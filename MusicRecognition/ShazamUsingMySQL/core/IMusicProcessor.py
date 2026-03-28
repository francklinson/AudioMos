import abc
import hashlib

import librosa
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage.filters import maximum_filter
from scipy.ndimage.morphology import generate_binary_structure, iterate_structure

from utils.hparam import hp


# 音乐的处理
class IMusicProcessor:

    # 创建指纹并保存到数据库中的接口
    def create_finger_prints_and_save_database(self, music_path, connector):
        raise NotImplementedError(u"出错了，你没有实现create_finger_prints_and_save_database抽象方法")

    def _calculation_hash(self, music_path):
        """
        计算指纹
        :param music_path: 音乐的路径
        :return: 指纹 [(hash,t1), (hash,t1)...]
        """
        # 音乐的预处理，转为频谱图(频谱矩阵)
        spectrogram = self._pre_music(music_path)
        # 处理频谱图
        spectrogram = self._spectrogram_handle(spectrogram)
        # 通过频谱图得到peakes
        peakes = self._fingerprint(spectrogram)
        # 通过peakes得到Hash并返回
        return self._generate_hash(peakes)

    @staticmethod
    def _pre_music(music_path):
        """
        音乐的预处理，转为频谱图(频谱矩阵)
        :param music_path: 音乐的路径
        :return: 频谱图
        """
        # 加载歌曲
        y, sr = librosa.load(music_path, sr=hp.fingerprint.core.stft.sr)
        # 做短时傅里叶变化
        arr2D = librosa.stft(y,
                             n_fft=hp.fingerprint.core.stft.n_fft,
                             hop_length=hp.fingerprint.core.stft.hop_length,
                             win_length=hp.fingerprint.core.stft.win_length
                             )
        # 返回的是（频率，时间）
        return np.abs(arr2D)

    # @cost_time
    @staticmethod
    def _spectrogram_handle(spectrogram):
        """
        处理频谱图
        :param spectrogram: 频谱图
        :return: 处理之后的频谱图
        """
        # 用最小值替换频谱矩阵的的0
        min_ = np.min(spectrogram[np.nonzero(spectrogram)])

        # 用得到的最小值替换全0
        spectrogram[spectrogram == 0] = min_

        # 取得log
        spectrogram = 10 * np.log10(spectrogram)

        # 防止数据为负无穷
        spectrogram[spectrogram == -np.inf] = 0

        # 返回处理之后的频谱图
        return spectrogram

    # 通过频谱图得到peakes
    # @cost_time
    def _fingerprint(self, spectrogram):
        """
        通过频谱图得到peakes
        :param spectrogram: 频谱图
        :return: 局部最大值点
        """
        # maximum_filter
        # 制作十字架
        struct = generate_binary_structure(2, 1)

        # 扩大十字架
        neighborhood = iterate_structure(struct, hp.fingerprint.core.neighborhood)

        # 取得局部最大值点
        local_max = maximum_filter(spectrogram, footprint=neighborhood) == spectrogram

        # 获取局部最大的能量值
        amps = spectrogram[local_max]

        # 拉平
        amps = amps.flatten()
        # 拿到局部最大值点的时间和频率两个轴的值，j表示频率，i表示时间
        j, i = np.where(local_max)

        # 得到（时间，频率，能量值）三元组数据即是我们要的peakes
        peakes = list(zip(i, j, amps))

        # 过滤能量小的值
        peakes = [item for item in peakes if item[2] > hp.fingerprint.core.amp_min]

        # 画图函数，星座图
        if hp.fingerprint.show_plot.create_database.planisphere_plot:
            self._draw_planisphere_plot(peakes)

        # 时间
        time_idx = [item[0] for item in peakes]

        # 频率
        freq_idx = [item[1] for item in peakes]

        # 包装起来
        peakes = list(zip(time_idx, freq_idx))

        return peakes

    # @cost_time
    @staticmethod
    def _generate_hash(peaks):
        """
        通过peakes得到Hash并返回
        :param peaks: 局部最大值点
        :return: Hash，[(hash,t1), (hash, t1), ]
        """
        # 按照时间进行排序
        peakes = sorted(peaks)

        # 遍历锚点
        for i in range(len(peakes)):
            # 遍历近邻点
            for j in range(1, hp.fingerprint.core.near_num):
                # 防止下标越界
                if i + j < len(peakes):
                    # 两个点的时间
                    t1 = peakes[i][0]
                    t2 = peakes[i + j][0]

                    # 两个点的频率
                    f1 = peakes[i][1]
                    f2 = peakes[i + j][1]

                    # 计算时间间隔
                    t_delta = t2 - t1
                    if hp.fingerprint.core.min_time_delta <= t_delta <= hp.fingerprint.core.max_time_delta:
                        # 计算Hash
                        hash_str = "%s|%s|%s" % (f1, f2, t_delta)
                        # 生成Hash
                        hash_str = hashlib.sha1(hash_str.encode("utf-8"))
                        yield hash_str.hexdigest(), t1

    # @cost_time
    @staticmethod
    def _draw_planisphere_plot(peaks):
        """
        绘制星座图
        :param peaks:
        :return:
        """
        x_and_y = [(item[1], item[0]) for item in peaks]

        # x坐标
        x = [int(item[0]) for item in x_and_y]
        # y坐标
        y = [int(item[1]) for item in x_and_y]

        plt.scatter(x, y, marker='x')
        plt.show()


class IMusicProcessorCreate(IMusicProcessor):

    # 创建指纹并保存到数据库中的接口
    @abc.abstractmethod
    def create_finger_prints_and_save_database(self, music_path, connector):
        raise NotImplementedError(u"出错了，你没有实现create_finger_prints_and_save_database抽象方法")


class IMusicProcessorPredict(IMusicProcessor):

    @abc.abstractmethod
    def predict_music(self, music_path, connector):
        raise NotImplementedError(u"出错了，你没有实现predict_music抽象方法")
