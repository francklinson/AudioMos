"""
@file: audio_mos.py
@time: 2025/7/2
@desc: 执行mos分计算
用到了带参考的计算方法：pesq、sisdr、stoi
无参考的计算方法：nisqa、dnsmos
基于wenet语音识别率的方法：wer

使用方法：
确保所有依赖库已安装。
将音频文件和参考文件放在est_dir和ref_dir目录下。
音频文件需要先经过audio_cut.py进行切分（根据1k标志音）、audio_align.py进行对齐，然后存放到est_dir目录下。
运行脚本：python audio_mos.py。

2026.3.2, 增加音色还原度评估方法
"""
import shutil
import wenet
from wenet.wer import wer
import os
import sys
import pandas as pd
from tqdm import tqdm
import soundfile as sf
import speechmetrics.speechmetrics as speechmetrics
import librosa
import numpy as np
import onnxruntime as ort
# from smoother import kalman_smoother # 暂时不知道为什么要卡尔曼滤波
import pesq
import warnings
from nisqa.predict import nisqa_predict
import torch.cuda as cuda
from modelscope import pipeline
import scoreq

warnings.filterwarnings("ignore")


def get_ref_file(input_wav_file, ref_dir):
    """
    获取与输入音频文件对应的参考文件。
    例如 : voice_mix_70dB_1_关_003.wav 对应于  ref_003.wav
    """
    ref_file_name = "ref_" + os.path.basename(input_wav_file).removesuffix('.wav').split('_')[-1] + '.wav'
    ref_file = os.path.join(ref_dir, ref_file_name)
    if os.path.exists(ref_file):
        return ref_file
    else:
        return None


def can_convert_to_float(x):
    """
    判断输入能否转换为浮点数
    """
    try:
        float(x)
        return True
    except ValueError:
        return False


def purge_all_file_cache(split_dir="split_out", test_dir="est_dir"):
    """
    清理所有缓存文件
    """
    file_cache_dir_list = [split_dir, test_dir]
    for file_cache_dir in file_cache_dir_list:
        if os.path.exists(file_cache_dir):
            print(f"正在清理缓存文件 {file_cache_dir}")
            shutil.rmtree(file_cache_dir)
            os.makedirs(file_cache_dir)
    print("缓存文件清理完成!")


class ToneColorFidelityScore:
    """
    基于人声embedding的相似度来评估测试音频对于参考音频（源音频）的音色还原程度
    基于modelscope提供的speaker verification模型实现
    """

    def __init__(self):
        """
        模型地址和对应的权重，权重是根据ERR值得到的，ERR越大，说明错误率越高，权重越低，反之亦然
        """
        self.sv_model_dict = {
            "campplus": {
                "path": "/mnt/test/scripts/speaker_verification/pretrained_models/speech_campplus_sv_zh-cn_16k-common/",
                "weight": 10 - 4.32},
            "ecapa-tdnn": {
                "path": "/mnt/test/scripts/speaker_verification/pretrained_models/speech_ecapa-tdnn_sv_zh-cn_cnceleb_16k/",
                "weight": 10 - 7.45},
            "eres2net": {
                "path": "/mnt/test/scripts/speaker_verification/pretrained_models/speech_eres2net_sv_zh-cn_16k-common/",
                "weight": 10 - 2.79},
            "eres2netv2": {
                "path": "/mnt/test/scripts/speaker_verification/pretrained_models/speech_eres2netv2_sv_zh-cn_16k-common/",
                "weight": 10 - 3.81},
            "res2net": {
                "path": "/mnt/test/scripts/speaker_verification/pretrained_models/speech_res2net_sv_zh-cn_3dspeaker_16k/",
                "weight": 10 - 5},
            "resnet34": {
                "path": "/mnt/test/scripts/speaker_verification/pretrained_models/speech_resnet34_sv_zh-cn_3dspeaker_16k/",
                "weight": 10 - 6.97}
        }

    @staticmethod
    def _compare_speakers(features1, features2):
        """
        计算余弦相似度
        """
        similarity = np.dot(features1, features2) / (
                np.linalg.norm(features1) * np.linalg.norm(features2)
        )
        return similarity

    def get_mos(self, input_test_file_list, ref_dir):
        """
        计算基于音色还原度的mos分
        """
        test_file_list = input_test_file_list.copy()
        # {file:{algorithm:{"embedding":embedding,"score":score}}}
        file_embedding_score_dict = dict()
        file_num = len(test_file_list)
        total_score_list = [0.0 for _ in range(file_num)]

        # 给待测音频注册一下
        for test_file in test_file_list:
            file_embedding_score_dict[test_file] = dict()

        # 给参考音频注册一下
        for ref_file in os.listdir(ref_dir):
            ref_file_full_path = os.path.join(ref_dir, ref_file)
            file_embedding_score_dict[ref_file_full_path] = dict()
            # 参考音频也加入待分析文件列表中
            test_file_list.append(ref_file_full_path)

        # 遍历算法
        for alg in self.sv_model_dict.keys():
            sv_pipeline = pipeline(
                task='speaker-verification',
                model=self.sv_model_dict[alg]["path"],
                model_revision='v1.0.0'
            )
            # 实际执行embedding计算
            result = sv_pipeline(test_file_list, output_emb=True)

            # 清除缓存，否则爆显存
            del sv_pipeline
            if cuda.is_available():
                cuda.empty_cache()
                cuda.synchronize()
            # 取结果中的'embs'字段
            all_embs = result['embs']
            for i in range(len(all_embs)):
                file_embedding_score_dict[test_file_list[i]][alg] = {"embedding": all_embs[i]}

        # 全部计算完了，进行结果的汇总

        # 遍历文件
        file_index = 0
        for file in test_file_list:
            ref_file = get_ref_file(file, ref_dir)
            # print("file:", file)
            # print("ref_file:", ref_file)
            file_total_score = 0
            if ref_file is not None and ref_file != file:
                # 遍历算法
                for alg in file_embedding_score_dict[file].keys():
                    # print(alg)
                    file_embedding_score_dict[file][alg]["score"] = self._compare_speakers(
                        file_embedding_score_dict[file][alg]["embedding"],
                        # 和参考文件的embedding计算相似度
                        file_embedding_score_dict[ref_file][alg]["embedding"])
                    # print("similarity:", file_embedding_score_dict[file][alg]["score"])
                    file_total_score += file_embedding_score_dict[file][alg]["score"] * self.sv_model_dict[alg][
                        "weight"]
                # 存储所有算法的加权结果
                # 默认返回np.float32，转一下float
                total_score_list[file_index] = float(file_total_score) / 30
                file_index += 1
            else:
                print("No proper reference audio file found!!!")
                file_index += 1
                continue

        return {"tcf": total_score_list}


class ScoreqScore:
    """
    基于scoreq的mos分计算
    参考：https://pypi.org/project/scoreq/
    """
    def __init__(self, data_domain='natural', mode='nr'):
        """
        初始化
        :param data_domain:
        """
        self.pred_mos_ins = scoreq.Scoreq(data_domain=data_domain, mode=mode,
                                          model_path="/mnt/test/scripts/label-studio-zch/label-studio-ml-backend/label_studio_ml/examples/voice_mos/scoreq_models/adapt_nr_telephone.onnx")

    def get_mos(self, file_dir_list):
        """
        计算mos分
        :param file_dir_list: 传入的音频列表
        """
        score_list = [0.0 for _ in range(len(file_dir_list))]
        file_index = 0
        for file in file_dir_list:
            try:
                pred_mos = self.pred_mos_ins.predict(file)
                score_list[file_index] = pred_mos
            except Exception as e:
                print(e)
            finally:
                file_index += 1
        return {"scoreq": score_list}


class DNSMOScore:
    def __init__(self) -> None:
        p808_model_path = os.path.join('DNSMOS', 'DNSMOS', 'model_v8.onnx')
        primary_model_path = os.path.join('DNSMOS', 'pDNSMOS', 'sig_bak_ovr.onnx')
        self.onnx_sess = ort.InferenceSession(primary_model_path)
        self.p808_onnx_sess = ort.InferenceSession(p808_model_path)
        self.INPUT_LENGTH = 9.01
        self.SAMPLING_RATE = 16000
        self.p808_model_path = os.path.join('DNSMOS', 'DNSMOS', 'model_v8.onnx')

    @staticmethod
    def __audio_melspec(audio, n_mels=120, frame_size=320, hop_length=160, sr=16000, to_db=True):
        """
        计算音频的梅尔频谱
        """
        mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_fft=frame_size + 1, hop_length=hop_length,
                                                  n_mels=n_mels)
        if to_db:
            mel_spec = (librosa.power_to_db(mel_spec, ref=np.max) + 40) / 40
        return mel_spec.T

    @staticmethod
    def __get_polyfit_val(sig, bak, ovr, is_personalized_MOS):
        """
        使用多项式拟合计算评分
        """
        if is_personalized_MOS:
            p_ovr = np.poly1d([-0.00533021, 0.005101, 1.18058466, -0.11236046])
            p_sig = np.poly1d([-0.01019296, 0.02751166, 1.19576786, -0.24348726])
            p_bak = np.poly1d([-0.04976499, 0.44276479, -0.1644611, 0.96883132])
        else:
            p_ovr = np.poly1d([-0.06766283, 1.11546468, 0.04602535])
            p_sig = np.poly1d([-0.08397278, 1.22083953, 0.0052439])
            p_bak = np.poly1d([-0.13166888, 1.60915514, -0.39604546])

        # print(sig)
        sig_poly = p_sig(sig)
        bak_poly = p_bak(bak)
        ovr_poly = p_ovr(ovr)

        return sig_poly, bak_poly, ovr_poly

    def __get_score(self, fpath, is_personalized_MOS=None):
        aud, input_fs = sf.read(fpath)
        fs = self.SAMPLING_RATE
        if input_fs != fs:
            audio = librosa.resample(aud, orig_sr=input_fs, target_sr=fs)
        else:
            audio = aud
        actual_audio_len = len(audio)
        len_samples = int(self.INPUT_LENGTH * fs)
        while len(audio) < len_samples:
            audio = np.append(audio, audio)

        num_hops = int(np.floor(len(audio) / fs) - self.INPUT_LENGTH) + 1
        hop_len_samples = fs
        predicted_mos_sig_seg_raw = []
        predicted_mos_bak_seg_raw = []
        predicted_mos_ovr_seg_raw = []
        predicted_mos_sig_seg = []
        predicted_mos_bak_seg = []
        predicted_mos_ovr_seg = []
        predicted_p808_mos = []

        for idx in range(num_hops):
            audio_seg = audio[int(idx * hop_len_samples): int((idx + self.INPUT_LENGTH) * hop_len_samples)]
            if len(audio_seg) < len_samples:
                continue

            input_features = np.array(audio_seg).astype('float32')[np.newaxis, :]
            p808_input_features = np.array(self.__audio_melspec(audio=audio_seg[:-160])).astype('float32')[
                np.newaxis, :, :]
            oi = {'input_1': input_features}
            p808_oi = {'input_1': p808_input_features}
            p808_mos = self.p808_onnx_sess.run(None, p808_oi)[0][0][0]
            mos_sig_raw, mos_bak_raw, mos_ovr_raw = self.onnx_sess.run(None, oi)[0][0]
            mos_sig, mos_bak, mos_ovr = self.__get_polyfit_val(mos_sig_raw, mos_bak_raw, mos_ovr_raw,
                                                               is_personalized_MOS)
            predicted_mos_sig_seg_raw.append(mos_sig_raw)
            predicted_mos_bak_seg_raw.append(mos_bak_raw)
            predicted_mos_ovr_seg_raw.append(mos_ovr_raw)
            predicted_mos_sig_seg.append(mos_sig)
            predicted_mos_bak_seg.append(mos_bak)
            predicted_mos_ovr_seg.append(mos_ovr)
            predicted_p808_mos.append(p808_mos)
        clip_dict = {'OVRL': np.mean(predicted_mos_ovr_seg), 'SIG': np.mean(predicted_mos_sig_seg),
                     'BAK': np.mean(predicted_mos_bak_seg), 'P808_MOS': np.mean(predicted_p808_mos)}
        return clip_dict

    def get_mos(self, file_list):
        """
        计算dnsmos
        """
        file_num = len(file_list)
        # 并发处理会打乱顺序 这里改成顺序处理
        ovrl = [0.0 for _ in range(file_num)]
        sig = [0.0 for _ in range(file_num)]
        bak = [0.0 for _ in range(file_num)]
        p808mos = [0.0 for _ in range(file_num)]
        file_index = 0
        for clip in tqdm(file_list, desc='DNSMOS'):
            if not clip.endswith(".wav"):
                file_index += 1
                continue
            try:
                data = self.__get_score(clip)
                ovrl[file_index] = float(data['OVRL'])
                sig[file_index] = float(data['SIG'])
                bak[file_index] = float(data['BAK'])
                p808mos[file_index] = float(data['P808_MOS'])
            except Exception as e:
                print(e)
            finally:
                file_index += 1
        res = {'OVRL': ovrl, 'SIG': sig, 'BAK': bak, 'P808_MOS': p808mos}
        return res


class WerScore:
    """
    用于计算基于语音识别率的WER评分。
    """

    def __init__(self):
        self.model = wenet.load_model(language="chinese",
                                      model_dir="/mnt/test/scripts/label-studio-zch/label-studio-ml-backend/label_studio_ml/examples/voice_mos/wenet/chinese/")

    @staticmethod
    def __get_ref_gt_text(input_wav_file):
        """
        获取ref的ground truth  text
        """
        ref_001_gt_text = '他为儿子买了一整根甘蔗市区的停车收费将大幅提高他醒来后发现自己脸上有黑眼圈'
        ref_002_gt_text = '大风刮倒了一处在建厂房姚大爷觉得车夫的想法蛮有道理汹涌的河水顺利而下流的很快'
        ref_003_gt_text = '坚持终于让他有所收获据说这是当地最古老的小区你就是那个爱打篮球的人'
        ref_004_gt_text = '总理对任何事情都要刨根问底渐渐的他还真就睡着了这身衣服就像被大雨淋过似的'
        input_file_name = os.path.basename(input_wav_file).removesuffix('.wav')
        if input_file_name.endswith('001'):
            return ref_001_gt_text
        elif input_file_name.endswith('002'):
            return ref_002_gt_text
        elif input_file_name.endswith('003'):
            return ref_003_gt_text
        elif input_file_name.endswith('004'):
            return ref_004_gt_text
        else:
            print("error: no ref gt text")
            return None

    def get_wer(self, file_dir_list):
        """
        计算wer
        """
        file_num = len(file_dir_list)
        # file_name = ["" for _ in range(file_num)]
        # res = [0.0 for _ in range(file_num)]
        wer_data = [0.0 for _ in range(file_num)]
        wcorr = [0.0 for _ in range(file_num)]
        file_index = 0
        for file in tqdm(file_dir_list, desc='asr'):
            if not file.endswith(".wav"):
                file_index += 1
                continue
            result = self.model.transcribe(file)
            try:
                ref = self.__get_ref_gt_text(file)
                tmp_wer, tmp_wcorr = wer(ref, result['text'])
                # file_name.append(file)
                # res.append(result['text'])
                wer_data[file_index] = tmp_wer
                wcorr[file_index] = tmp_wcorr
            except Exception as e:
                print(e)
            finally:
                file_index += 1

        data = {'wer': wer_data, 'wcorr': wcorr}
        return data


class RefScore:
    """
    用于计算带参考的音频质量评分（STOI, SISDR, PESQ）
    """

    def __init__(self):
        self.metrics = speechmetrics.load(["stoi", "sisdr"])

    @staticmethod
    def pesq_to_mos_lqo(pesq_score):
        """
        pesq分到mos_lqo分的映射
        """
        return 0.999 + 4 / (1 + np.exp(-1.4945 * pesq_score + 4.6607))

    def get_mos(self, file_list, ref_dir):
        """
        计算stoi、sisdr、pesq分
        """
        file_num = len(file_list)
        file_names = ["" for _ in range(file_num)]
        STOI = [0.0 for _ in range(file_num)]
        SISDR = [0.0 for _ in range(file_num)]
        PESQ = [0.0 for _ in range(file_num)]
        file_index = 0
        for file in tqdm(file_list, desc='speechmetrics'):
            if not file.endswith(".wav"):
                file_index += 1
                continue
            path_to_reference = get_ref_file(file, ref_dir)
            try:
                # stoi 和 sisdr
                scores = self.metrics(file, path_to_reference)
                file_names[file_index] = file
                STOI[file_index] = scores['stoi'].mean()
                SISDR[file_index] = scores['sisdr'].mean()
                # pesq
                ref, sr_ref = sf.read(path_to_reference)
                est, sr_est = sf.read(file)
                if sr_est != 16000:
                    est = librosa.resample(est, orig_sr=sr_est, target_sr=16000)
                if sr_ref != 16000:
                    ref = librosa.resample(ref, orig_sr=sr_ref, target_sr=16000)
                tmp = pesq.pesq(fs=16000, ref=ref, deg=est, mode='wb')

                PESQ[file_index] = tmp
                # PESQ.append(self.pesq_to_mos_lqo(tmp))
            except Exception as e:
                print(e)
            finally:
                file_index += 1

        data = {
            'STOI': STOI,
            'SISDR': SISDR,
            'pesq': PESQ}
        return data


class NisqaMosScore:
    def __init__(self):
        self.nisqa_mode = "predict_list"
        self.nisqa_model = 'nisqa_3000.tar'

    def get_mos(self, file_dir_list) -> dict:
        """
        计算nisqa分
        """
        file_num = len(file_dir_list)
        nisqa_prediction: pd.DataFrame = nisqa_predict(mode=self.nisqa_mode, deg_list=file_dir_list,
                                                       model=self.nisqa_model)
        ret = nisqa_prediction.to_dict(orient='list')
        ret.pop("deg")
        # ret的格式：
        # {'mos_pred': [2.5582544803619385, 1.9623147249221802, 2.091243028640747],
        #  'noi_pred': [3.740171432495117, 3.429931163787842, 3.5098958015441895],
        #  'dis_pred': [2.727465867996216, 2.2863333225250244, 2.0204949378967285],
        #  'col_pred': [2.3999359607696533, 2.124859094619751, 2.1804261207580566],
        #  'loud_pred': [3.2005224227905273, 2.667280912399292, 2.8841376304626465]}
        # 验证一下输出长度
        for k, v in ret.items():
            if len(v) != file_num:
                print(f"Length of {k}: {len(v)} does not match length of input, please check!!!")
                # 补0
                ret[k].extend([0.0 for _ in range(file_num - len(v))])
        return ret


def main(input_dir, ref_dir):
    """
    主程序逻辑，调用各个评分类计算评分，并将结果保存到Excel文件中。
    """
    # result_file = open(os.path.join(os.getcwd(), 'result.txt'), 'w')
    # sys.stdout = result_file
    data = {}
    # 获取输入目录中所有.wav文件的文件名列表和完整路径列表
    file_name_list = [file for file in os.listdir(input_dir) if file.endswith('.wav')]
    file_dir_list = [os.path.join(input_dir, file) for file in file_name_list]

    data.update({'文件名': file_name_list})

    "stoi、sisdr、pesq"
    ref_score_ins = RefScore()
    ref_scores = ref_score_ins.get_mos(file_dir_list, ref_dir)
    data.update(ref_scores)

    "dnsmos"
    dnsmos_calc_ins = DNSMOScore()
    dnsmos_scores = dnsmos_calc_ins.get_mos(file_dir_list)
    data.update(dnsmos_scores)

    "wer"
    wer_calc_ins = WerScore()
    wer_scores = wer_calc_ins.get_wer(file_dir_list)
    data.update(wer_scores)

    "nisqa_dim"
    nisqa_calc_ins = NisqaMosScore()
    nisqa_dim_scores = nisqa_calc_ins.get_mos(file_dir_list)
    data.update(nisqa_dim_scores)

    "音色还原度"
    # tcf_ins = ToneColorFidelityScore()
    # tcf_scores = tcf_ins.get_mos(file_dir_list, ref_dir)
    # data.update(tcf_scores)

    # result_file.close()
    values = np.array(list(data.values())).T
    final_scores = []
    for i in range(len(values)):
        val_tmp = [float(s) if can_convert_to_float(s) else s for s in values[i]]
        # 加权 求最终分数
        tmp = np.mean([val_tmp[1] * 5, (1 / (1 + np.exp(-val_tmp[2]))) * 5, val_tmp[3],
                       val_tmp[4], val_tmp[5], val_tmp[7], (1 - val_tmp[8]) * 5, val_tmp[9] * 5,
                       val_tmp[10], val_tmp[11], val_tmp[12], val_tmp[13], val_tmp[14], ])
        # val_tmp[15] * 5, ])
        final_scores.append(tmp)
    data.update({'final_scores': final_scores})
    df = pd.DataFrame(data)
    out_excel_name = 'MOS结果汇总.xlsx'
    df.to_excel(out_excel_name, index=False, )
    # sys.stdout = sys.__stdout__
    print(f'评分结果已保存至 {out_excel_name}')


if __name__ == '__main__':
    main("/mnt/test/scripts/label-studio-zch/label-studio-ml-backend/label_studio_ml/examples/voice_mos/est_dir/",
         "/mnt/test/scripts/label-studio-zch/label-studio-ml-backend/label_studio_ml/examples/voice_mos/ref_dir/")
    # main("est_dir", "ref_dir")
    # purge_all_file_cache()
