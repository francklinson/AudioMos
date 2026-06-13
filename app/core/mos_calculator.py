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
import os
import sys

# 添加本地包路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'algorithms', 'speechmetrics'))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'algorithms'))  # nisqa的父目录
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'algorithms', 'wenet'))

import pandas as pd
from tqdm import tqdm
import soundfile as sf

# 尝试导入speechmetrics
try:
    import speechmetrics.speechmetrics as speechmetrics
except ImportError:
    try:
        import speechmetrics
    except ImportError:
        speechmetrics = None
        print("警告: speechmetrics未安装，STOI/SISDR评分将不可用")

import librosa
import numpy as np
import onnxruntime as ort
import warnings
import torch.cuda as cuda

# 尝试导入scoreq
try:
    import scoreq
except ImportError:
    scoreq = None
    print("警告: scoreq未安装，Scoreq评分将不可用")

warnings.filterwarnings("ignore")

# 尝试导入pesq，如果失败则提供替代实现
try:
    import pesq
    PESQ_AVAILABLE = True
except ImportError:
    PESQ_AVAILABLE = False
    print("警告: pesq模块未安装，PESQ评分将不可用")

# 尝试导入nisqa
try:
    from nisqa.predict import nisqa_predict
    NISQA_AVAILABLE = True
except ImportError as e:
    NISQA_AVAILABLE = False
    print(f"警告: nisqa模块未安装，NISQA评分将不可用 - {e}")

# 尝试导入wenet（完整的语音识别库）
WENET_AVAILABLE = False
try:
    # 添加wenet路径
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'algorithms', 'wenet'))
    # 导入wenet语音识别库
    import wenet
    if hasattr(wenet, 'load_model'):
        # 导入项目本地的wer模块用于WER计算
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'algorithms', 'wenet', 'wenet_local'))
        from wer import wer
        WENET_AVAILABLE = True
    else:
        print("警告: wenet库不完整（缺少load_model），WER评分将不可用")
except ImportError as e:
    print(f"警告: wenet模块未安装或导入失败，WER评分将不可用 - {e}")

# ModelScope兼容补丁（解决datasets版本兼容性问题）
try:
    import datasets
    if not hasattr(datasets, 'LargeList'):
        class _LargeListStub(list):
            pass
        datasets.LargeList = _LargeListStub
        print("已应用ModelScope兼容补丁 (datasets.LargeList)")
except ImportError:
    pass

# 尝试导入modelscope
try:
    from modelscope import pipeline
    MODELSCOPE_AVAILABLE = True
    print("✓ ModelScope可用")
except ImportError as e:
    MODELSCOPE_AVAILABLE = False
    print(f"警告: modelscope模块未安装或导入失败，音色还原度评分将不可用 - {e}")


def get_ref_file(input_wav_file, ref_dir):
    """
    获取与输入音频文件对应的参考文件。
    支持多种匹配方式：
    例如 : voice_mix_70dB_1_关_003.wav 对应于  ref_003.wav
    """
    input_basename = os.path.basename(input_wav_file)
    input_name = input_basename.removesuffix('.wav')

    # 方式1: 直接匹配相同文件名
    ref_file = os.path.join(ref_dir, input_basename)
    if os.path.exists(ref_file):
        return ref_file

    # 方式2: 匹配 ref_前缀 + 完整文件名
    ref_file_name = "ref_" + input_basename
    ref_file = os.path.join(ref_dir, ref_file_name)
    if os.path.exists(ref_file):
        return ref_file

    # 方式3: 从文件名提取ID匹配 (如 test_001.wav -> ref_001.wav)
    parts = input_name.split('_')
    if len(parts) >= 2:
        file_id = parts[-1]
        ref_file_name = f"ref_{file_id}.wav"
        ref_file = os.path.join(ref_dir, ref_file_name)
        if os.path.exists(ref_file):
            return ref_file

    # 方式4: 尝试匹配不带后缀的文件名 (如 test.wav -> ref_test.wav)
    ref_file_name = f"ref_{input_name}.wav"
    ref_file = os.path.join(ref_dir, ref_file_name)
    if os.path.exists(ref_file):
        return ref_file

    return None


def can_convert_to_float(x):
    """
    判断输入能否转换为浮点数
    """
    try:
        float(x)
        return True
    except (ValueError, TypeError):
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
        if not MODELSCOPE_AVAILABLE:
            raise ImportError("modelscope未安装，无法使用音色还原度评分")

        # 使用ModelScope模型ID代替本地路径
        # 模型会自动下载到 ~/.cache/modelscope/hub/
        # 只使用一个最稳定的模型
        self.sv_model_dict = {
            "eres2net": {
                "path": "damo/speech_eres2net_sv_zh-cn_16k-common",
                "weight": 1.0}
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
        # 只处理原始输入文件（前file_num个），不包括参考文件
        for file_index in range(file_num):
            file = input_test_file_list[file_index]
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
            else:
                print(f"No proper reference audio file found for {file}!!!")

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
        if scoreq is None:
            raise ImportError("scoreq未安装，无法使用Scoreq评分")
        
        # Scoreq使用ONNX Runtime，模型会自动管理
        # 不需要指定本地模型路径
        self.pred_mos_ins = scoreq.Scoreq(data_domain=data_domain, mode=mode)

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
        # 获取项目根目录
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        # 优先检查 models/dnsmos/，然后检查 app/algorithms/dnsmos/
        p808_model_path = os.path.join(project_root, 'models', 'dnsmos', 'DNSMOS', 'model_v8.onnx')
        primary_model_path = os.path.join(project_root, 'models', 'dnsmos', 'pDNSMOS', 'sig_bak_ovr.onnx')
        if not os.path.exists(p808_model_path):
            p808_model_path = os.path.join(project_root, 'app', 'algorithms', 'dnsmos', 'DNSMOS', 'model_v8.onnx')
            primary_model_path = os.path.join(project_root, 'app', 'algorithms', 'dnsmos', 'pDNSMOS', 'sig_bak_ovr.onnx')
        self.onnx_sess = ort.InferenceSession(primary_model_path)
        self.p808_onnx_sess = ort.InferenceSession(p808_model_path)
        self.INPUT_LENGTH = 9.01
        self.SAMPLING_RATE = 16000
        self.p808_model_path = p808_model_path

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
            try:
                data = self.__get_score(clip)
                ovrl[file_index] = float(data['OVRL'])
                sig[file_index] = float(data['SIG'])
                bak[file_index] = float(data['BAK'])
                p808mos[file_index] = float(data['P808_MOS'])
            except Exception as e:
                print(f"DNSMOS计算失败 {clip}: {e}")
            file_index += 1
        res = {'OVRL': ovrl, 'SIG': sig, 'BAK': bak, 'P808_MOS': p808mos}
        return res


class WerScore:
    """
    用于计算基于语音识别率的WER评分。
    """

    def __init__(self):
        if not WENET_AVAILABLE:
            raise ImportError("wenet未安装，无法使用WER评分")
        # 使用本地模型路径或预训练模型名称
        # 优先使用 wenetspeech 模型（通过Hub下载）
        model_path = os.path.expanduser("~/.wenet/wenetspeech")
        if os.path.exists(model_path):
            print(f"使用本地模型: {model_path}")
            self.model = wenet.load_model(model_path)
        else:
            # 使用预训练模型名称（会自动下载）
            print("下载并使用 wenetspeech 模型...")
            self.model = wenet.load_model("wenetspeech")

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
            try:
                result = self.model.transcribe(file)
                ref = self.__get_ref_gt_text(file)
                # 处理 DecodeResult 对象，获取 text 属性
                if hasattr(result, 'text'):
                    text = result.text
                elif isinstance(result, dict):
                    text = result['text']
                else:
                    text = str(result)
                tmp_wer, tmp_wcorr = wer(ref, text)
                # file_name.append(file)
                # res.append(result['text'])
                wer_data[file_index] = tmp_wer
                wcorr[file_index] = tmp_wcorr
            except Exception as e:
                print(f"WER计算失败 {file}: {e}")
            file_index += 1

        data = {'wer': wer_data, 'wcorr': wcorr}
        return data


class RefScore:
    """
    用于计算带参考的音频质量评分（STOI, SISDR, PESQ）
    """

    def __init__(self):
        if speechmetrics is None:
            raise ImportError("speechmetrics未安装，无法使用STOI/SISDR评分")
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
        
        print(f"\n[RefScore-原版] 开始计算，文件数: {file_num}, 参考目录: {ref_dir}")
        
        for file in tqdm(file_list, desc='speechmetrics'):
            file_basename = os.path.basename(file)
            path_to_reference = get_ref_file(file, ref_dir)
            
            if path_to_reference is None:
                print(f"[RefScore-原版] ⚠️ 未找到参考文件 for {file_basename}")
                file_index += 1
                continue
            
            print(f"[RefScore-原版] 处理 {file_basename} -> ref: {os.path.basename(path_to_reference)}")
            
            try:
                # stoi 和 sisdr
                scores = self.metrics(file, path_to_reference)
                file_names[file_index] = file
                stoi_val = scores['stoi'].mean()
                sisdr_val = scores['sisdr'].mean()
                STOI[file_index] = stoi_val
                SISDR[file_index] = sisdr_val
                print(f"[RefScore-原版]   ✓ STOI={stoi_val:.4f}, SISDR={sisdr_val:.4f}")
                
                # pesq
                if PESQ_AVAILABLE:
                    ref, sr_ref = sf.read(path_to_reference)
                    est, sr_est = sf.read(file)
                    if sr_est != 16000:
                        est = librosa.resample(est, orig_sr=sr_est, target_sr=16000)
                    if sr_ref != 16000:
                        ref = librosa.resample(ref, orig_sr=sr_ref, target_sr=16000)
                    tmp = pesq.pesq(fs=16000, ref=ref, deg=est, mode='wb')
                    PESQ[file_index] = tmp
                    print(f"[RefScore-原版]   ✓ PESQ={tmp:.4f}")
                else:
                    PESQ[file_index] = 0.0
            except Exception as e:
                print(f"[RefScore-原版] ❌ 计算失败 {file_basename}: {e}")
                import traceback
                print(f"[RefScore-原版] 错误详情: {traceback.format_exc()}")
            file_index += 1

        print(f"\n[RefScore-原版] 计算完成 - STOI: {STOI}, SISDR: {SISDR}, PESQ: {PESQ}")
        data = {
            'STOI': STOI,
            'SISDR': SISDR,
            'pesq': PESQ}
        return data


class NisqaMosScore:
    def __init__(self):
        if not NISQA_AVAILABLE:
            raise ImportError("nisqa未安装，无法使用NISQA评分")
        self.nisqa_mode = "predict_list"
        # 使用支持更长音频的模型文件 (ms_max_segments=3000, 支持~120秒音频)
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
    if WENET_AVAILABLE:
        wer_calc_ins = WerScore()
        wer_scores = wer_calc_ins.get_wer(file_dir_list)
        data.update(wer_scores)
    else:
        data.update({'wer': [0.0] * len(file_dir_list), 'wcorr': [0.0] * len(file_dir_list)})

    "nisqa_dim"
    if NISQA_AVAILABLE:
        nisqa_calc_ins = NisqaMosScore()
        nisqa_dim_scores = nisqa_calc_ins.get_mos(file_dir_list)
        data.update(nisqa_dim_scores)
    else:
        data.update({'mos_pred': [0.0] * len(file_dir_list), 'noi_pred': [0.0] * len(file_dir_list),
                     'dis_pred': [0.0] * len(file_dir_list), 'col_pred': [0.0] * len(file_dir_list),
                     'loud_pred': [0.0] * len(file_dir_list)})

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
