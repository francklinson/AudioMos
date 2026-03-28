import os
import numpy as np
import soundfile as sf
import librosa
import math
import statistics
from matplotlib import pyplot as plt
from mosqito import loudness_zwtv, roughness_dw, sharpness_din_tv, pr_ecma_st, pr_ecma_perseg, tnr_ecma_st, \
    tnr_ecma_perseg
from spafe.features import bfcc


class NoiseMos:
    def __init__(self, tonality_method="tnr", tonality_feature="stationary"):
        """
        初始化NoiseMos类。
        tonality_method：音调方法，可选值为"tnr"或"pr"。默认为"tnr"。
        tonality_feature：音调特征，可选值为"stationary"或"non-stationary"。默认为"stationary"。
        """
        assert tonality_method in ["tnr", "pr"]
        assert tonality_feature in ["stationary", "non-stationary"]
        self.tonality_method = tonality_method
        self.tonality_feature = tonality_feature
        self.audio = None
        self.fs = None

    @staticmethod
    def _get_noise_rms(target_file, offset=0.0, target_length=None):
        """
        计算音频文件中噪声的RMS。
        target_file：目标音频文件的路径。
        offset：音频文件的起始偏移量，单位为秒。默认为0.0。
        target_length：音频文件的持续时间，单位为秒。默认为None，表示读取整个文件。
        """
        audio, sr = librosa.load(target_file, offset=offset, duration=target_length)
        # 用librosa读取，数据为-1~1之间的范围，无需再除以32767
        temp = (audio ** 2).mean()
        if temp == 0.0:
            temp = 1e-15
        noise_rms = 10 * math.log(temp, 10)
        # score_rms = -5*math.tanh(max((noise_rms-(-45))/20,0))
        # print(noise_rms,score_rms)
        return noise_rms

    def _get_noise_var(self, target_file, offset=0.0, duration=None, bin_length=1.0):
        """
        计算音频文件中噪声的方差。
        target_file：目标音频文件的路径。
        offset：音频文件的起始偏移量，单位为秒。默认为0.0。
        duration：音频文件的持续时间，单位为秒。默认为None，表示读取整个文件。
        bin_length：每个时间段的长度，单位为秒。默认为1.0秒。
        """
        audio, sr = librosa.load(target_file, offset=offset, duration=duration)
        rms_list = []
        bin_number = int(len(audio) / sr // bin_length)
        for i in range(0, bin_number):
            tmp_offset = offset + bin_length * i
            tmp_duration = bin_length
            tmp_noise_rms = self._get_noise_rms(target_file, tmp_offset, tmp_duration)
            rms_list.append(tmp_noise_rms)
        noise_var = statistics.stdev(rms_list)
        # print(noise_var)
        # M=2
        # score_var = -5*math.tanh(noise_var/M)
        return noise_var


    def _get_noise_comfort(self, target_file, noise_file="noise/pink_noise.wav", offset=0.0, target_length=None):
        """
        生成一个与目标音频文件噪声部分具有相似特性的粉红噪声，并计算该粉红噪声与目标噪声之间的相似度得分。
        """
        # 计算噪声部分的RMS
        noise_rms = self._get_noise_rms(target_file, offset, target_length)
        noise, sr_noise = sf.read(target_file)
        length = len(noise) / sr_noise

        # 计算噪声的BFCC，调用spafe库
        noise_bfcc = bfcc.bfcc(noise,sr_noise)

        # 读取粉噪声
        pink_noise, sr_pink = sf.read(noise_file)
        audio_temp = pink_noise[0:int(length * sr_pink)]

        # 改变粉噪声的RMS
        audio_temp_rms = 10 * math.log((audio_temp ** 2).mean(), 10)
        multiply_coe = pow(10, (noise_rms - audio_temp_rms) / 10)
        # print(noise_rms, audio_temp_rms)
        audio_final = audio_temp * multiply_coe
        temp_file = 'pink_temp.wav'
        sf.write(temp_file, audio_final, samplerate=sr_pink)

        # 计算粉噪声的BFCC
        audio_bfcc = bfcc.bfcc(audio_final,sr_pink)
        # 计算粉噪声和目标噪声的BFCC的协方差
        result = np.corrcoef(audio_bfcc, noise_bfcc)[0][1]

        # 计算噪声频谱分布的得分
        score = -5 * math.tanh(np.max(result, 0))
        return result, score

    @staticmethod
    def _acoustic_fluctuation(specificLoudness, fmod=4):
        """
        计算声学波动性
        """
        specific_loudness_diff = np.zeros(len(specificLoudness))
        for i in range(len(specificLoudness)):
            if i == 0:
                specific_loudness_diff[i] = specificLoudness[i]
            else:
                specific_loudness_diff[i] = abs(specificLoudness[i] - specificLoudness[i - 1])
        F = (0.008 * sum(0.1 * specific_loudness_diff)) / ((fmod / 4) + (fmod / 4))
        return F

    def _load_audio(self, audio_file):
        """
        加载音频文件
        """
        # 检查文件是否存在以及是否是wav文件
        assert os.path.exists(audio_file), "File not found: " + audio_file
        assert audio_file.endswith('.wav'), "File format not supported, use .wav instead "
        self.audio = None
        self.fs = None
        # 加载音频文件
        self.audio, self.fs = librosa.load(audio_file)

    def _get_zwicker_loudness(self):
        """
        计算Zwicker响度
        """
        # quite slow
        N, N_spec, bark_axis, time_axis = loudness_zwtv(self.audio, self.fs)
        N_sort = np.sort(N)
        # 响度用N5计算，是在测量时间的5%内达到或超过的响度。
        # 这意味着 N5 代表一个接近噪声的响度-时间函数最大值的响度值。由于sort后是按照从小到大排序，因此为95%
        N_5 = N_sort[int(len(N_sort) * 0.95)]
        # print("N_5: ", N_5)
        # 避免响度值为0
        if N_5 == 0:
            N_5 = 0.000001
            print("N_5 is 0, set to 0.000001")
        return N_5, N, N_spec, bark_axis, time_axis

    def _get_tonality(self, plot_tonality=False):
        """
        计算音调度
        """
        tonality = None
        time = None
        print(f"Tonality setting: {self.tonality_feature}, {self.tonality_method}")
        if self.tonality_feature == "stationary":
            if self.tonality_method == "pr":
                tonality, _, _, _ = pr_ecma_st(self.audio, fs=self.fs, )
            elif self.tonality_method == "tnr":
                tonality, _, _, _ = tnr_ecma_st(self.audio, fs=self.fs)
        elif self.tonality_feature == "non-stationary":
            if self.tonality_method == "pr":
                tonality, _, _, _, time = pr_ecma_perseg(self.audio, fs=self.fs, )
            elif self.tonality_method == "tnr":
                tonality, _, _, _, time = tnr_ecma_perseg(self.audio, fs=self.fs)
        if plot_tonality is True:
            plt.plot(time, tonality)
            plt.xlabel("Time (s)")
            plt.ylabel("Tonality")
            plt.show()
        return tonality, time

    def cal_noise_mos(self, input_noise_file):
        """
        计算噪声的MOS值，并将结果保存到Excel文件中。
        根据Zwicker心理声学模型给出的噪声的心理声学烦恼度与其响度、粗糙度、尖锐度、波动度的关系进行计算。
        参考GB/T 42473-2023, 噪声主观质量评价方法, 附录D
        """
        # 加载数据
        print("Processing: ", input_noise_file)
        self._load_audio(input_noise_file)

        # 计算rms
        noise_rms = self._get_noise_rms(input_noise_file)
        # print("noise_rms: ", noise_rms)

        # 计算var
        noise_var = self._get_noise_var(input_noise_file, bin_length=1.0)
        # print("var: ", noise_var)

        # 计算舒适度
        comfort, comfort_score = self._get_noise_comfort(input_noise_file)

        # 计算roughness 粗糙度
        R, R_specific, bark, time = roughness_dw(self.audio, fs=self.fs, overlap=0)
        roughness = np.mean(R)
        # print("roughness: ", roughness)

        # 计算Zwicker响度
        N_5, N, N_spec, bark_axis, time_axis = self._get_zwicker_loudness()

        # 计算波动度
        fluctuation = self._acoustic_fluctuation(N)
        # print("fluctuation: ", fluctuation)

        # 计算表征波动和粗糙度的系数
        wfr = 2.18 * (0.6 * roughness + 0.4 * fluctuation) / np.power(N_5, 0.4)
        # wfr_without_fluc = 2.18 * (0.6 * roughness) / np.power(N_5, 0.4)
        # print("wfr_with_fluc: ", wfr_with_fluc)
        # print("wfr_without_fluc: ", wfr_without_fluc)

        # 计算sharpness，尖锐度
        S, time_ax = sharpness_din_tv(self.audio, fs=self.fs, skip=0)
        sharpness = np.mean(S)
        # print("sharpness: ", sharpness)

        # 计算表征尖锐度的系数
        if sharpness > 1.75:
            ws = (sharpness - 1.75) * 0.25 * np.log10(N_5 + 10)
        else:
            ws = 0

        # 计算音调度
        tonality, t_time = self._get_tonality(plot_tonality=False)

        # print("tonality: ", tonality)
        wt = 6.41 / np.power(N_5, 0.52) * tonality
        # print("wt: ", wt)

        # 计算Ap
        Ap = N_5 * (1 + np.power(np.power(ws, 2) + np.power(wfr, 2), 0.5))
        Ap2 = N_5 * (1 + np.power(np.power(ws, 2) + np.power(wfr, 2) + np.power(wt, 2), 0.5))

        return {"noise_rms": noise_rms, "noise_var": noise_var, "roughness": roughness,
                "fluctuation": fluctuation, "sharpness": sharpness, "Ap": Ap, "Ap2": Ap2}


if __name__ == '__main__':
    c = NoiseMos(tonality_feature="non-stationary", tonality_method="tnr")
    # print(c.cal_noise_mos(input_noise_file="noise/1.wav"))
    # print(c.cal_noise_mos(input_noise_file="noise/2.wav"))
    print(c.cal_noise_mos(input_noise_file="noise/3.wav"))
