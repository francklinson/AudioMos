import time
import torchaudio
import torch
import numpy as np
from tqdm import trange

'''
Original code: https://github.com/stephencwelch/Perceptual-Coding-In-Python/tree/master/PEAQPython
PEAQ(Perceptual Evaluation of Audio Quality)音频质量评估系统的Python实现。
PEAQ是ITU-R BS.1387标准中定义的客观音频质量评估方法，用于评估音频压缩算法或传输系统对音频质量的影响。
'''


class PQEval(object):
    """
    实现了PEAQ算法的核心功能
    包含音频信号预处理、频谱分析、心理声学模型计算等功能
    主要方法包括：
        PQDFTFrame：对音频帧进行DFT变换
        PQ_excitCB：计算临界频带激励
        PQgroupCB：将DFT能量分组到临界频带
        PQspreadCB：计算频域扩散
        PQ_timeSpread：计算时域扩散
        PQloud：计算响度值
        PQmovPD：计算概率检测

    """

    def __init__(self, Amax=1, Fs=48000, NF=2048):
        """
        初始化函数，用于设置音频处理所需的各种参数和预计算量。
        参数:
            Amax (float): 信号的最大振幅，默认值为1
            Fs (int): 采样频率，默认值为48000Hz
            NF (int): FFT点数，默认值为2048 (为了加快计算，改成4096)
        """
        # Amax is maximum signal amplitude, Fs is sampling frequency
        # Setup parameters and precompute quantities we'll need.
        self.Fs = Fs  # 采样频率
        self.NF = NF  # FFT点数

        # Hardcode the louness scalling params:
        fcLoudness = 1019.5  # 响度计算的中心频率
        Lp = 92  # 响度参数

        # Set up the window (including all gains)
        self.GL = self.PQ_GL(NF=self.NF, Amax=Amax, fcN=fcLoudness / self.Fs, Lp=Lp)  # 计算窗口增益

        # Precompute hann window:
        self.hw = self.GL * self.PQHannWin(self.NF)  # 计算汉宁窗口

        # Precompute frequency vector:
        self.f = np.linspace(0, self.Fs // 2, self.NF // 2 + 1)  # 频率向量

        # Outer and middle ear weighting:
        self.W2 = self.PQWOME(self.f)  # 外耳和中耳加权

        # Critical band constants:
        self.Nc, self.fc, self.fl, self.fu, self.dz = self.PQCB()  # 临界带常数

        # Internal Noise:
        self.EIN = self.PQIntNoise(self.fc)  # 内部噪声

        # Precompute normalization for frequency spreading:
        self.Bs = self.PQ_SpreadCB(np.ones(self.Nc), np.ones(self.Nc))  # 频率扩散归一化

        # Allocate storage
        self.Eb = np.zeros((2, self.Nc))  # 存储能量带
        self.Xw2 = np.zeros((2, self.NF // 2 + 1))  # 存储加窗后的频谱
        self.XwN2 = np.zeros(self.NF // 2 + 1)  # 存储归一化的加窗频谱
        self.E = np.zeros(self.Eb.shape)  # 存储能量
        self.Es = np.zeros((2, self.Nc))  # 存储扩展能量

        # Precompute for PQ Group:
        self.df = float(self.Fs) / self.NF  # 频率分辨率
        self.Emin = 1e-12  # 最小能量阈值

        self.U = np.zeros((self.NF // 2 + 1, self.Nc))  # 存储频率-临界带矩阵

        for k in range(self.NF // 2 + 1):
            for i in range(self.Nc):
                # 计算每个频率点在每个临界带中的重叠部分
                temp = (np.amin([self.fu[i], (k + 0.5) * self.df]) - np.amax(
                    [self.fl[i], (k - 0.5) * self.df])) / self.df
                self.U[k, i] = np.amax([0, temp])  # 确保非负

        # check FLAG, False means first operation
        self.check_PQmodPatt = False  # 模式检查标志，False表示首次操作

    def PQDFTFrame(self, x):
        # Window the data
        xw = self.hw * x

        # DFT (output is real followed by imaginary)
        X = self.PQRFFT(xw, self.NF, 1)

        # Squared magnitude
        X2 = self.PQRFFTMSq(X, self.NF)

        return X2

    def PQ_excitCB(self, X2):
        # Critical band grouping and frequency spreading

        # Outer and middle ear filtering
        self.Xw2[0, :] = self.W2 * X2[0, 0:self.NF // 2 + 1]
        self.Xw2[1, :] = self.W2 * X2[1, 0:self.NF // 2 + 1]

        # Form the difference magnitude signal
        self.XwN2 = self.Xw2[0, :] - 2 * np.sqrt(self.Xw2[0, :] * self.Xw2[1, :]) + self.Xw2[1, :]

        # Group into partial critical bands
        self.Eb[0, :] = self.PQgroupCB(self.Xw2[0, :])
        self.Eb[1, :] = self.PQgroupCB(self.Xw2[1, :])
        self.EbN = self.PQgroupCB(self.XwN2)

        # Add the internal noise term => "Pitch patterns"
        self.E[0, :] = self.Eb[0, :] + self.EIN
        self.E[1, :] = self.Eb[1, :] + self.EIN

        # Critical band spreading => "Unsmeared (in time) excitation patterns"
        self.Es[0, :] = self.PQspreadCB(self.E[0, :])
        self.Es[1, :] = self.PQspreadCB(self.E[1, :])

        return self.EbN, self.Es

    def PQgroupCB(self, X2):
        # Group a DFT energy vector into critical bands
        # X2 - Squared-magnitude vector (DFT bins)
        # Eb - Excitation vector (fractional critical bands)

        Eb = np.dot(X2, self.U)
        Eb[Eb < self.Emin] = self.Emin

        return Eb

    def PQspreadCB(self, E):
        # Spread an excitation vector (pitch pattern) - FFT model
        # Both E and Es are powers	    
        Es = self.PQ_SpreadCB(E, self.Bs)

        return Es

    def PQ_SpreadCB(self, E, Bs):
        e = 0.4  # Commonly used power value

        # Initialize arrays for storage. These values are used
        # in each iteration (summed over, multiplied, raised to
        # powers, etc.) when computing the spread Bark-domain
        # energy Es.
        #
        # aUCEe is for the product of bin-dependent (index l)
        # term aC, energy-dependent (E) term aE, and
        # term aU.
        #
        # Ene is (E[l]/A(l,E[l]))^e, stored for each index l
        #
        # Es is the overall spread Bark-domain energy
        #

        aUCEe = np.zeros(self.Nc)
        Ene = np.zeros(self.Nc)
        Es = np.zeros(self.Nc)

        # Calculate energy-dependent terms
        aL = 10 ** (2.7 * self.dz)

        for l in range(self.Nc):
            aUC = 10 ** ((-2.4 - 23 / self.fc[l]) * self.dz)
            aUCE = aUC * (E[l] ** (0.2 * self.dz))
            gIL = (1 - aL ** (-1 * (l + 1))) / (1 - aL ** (-1))
            gIU = (1 - (aUCE) ** (self.Nc - l)) / (1 - aUCE)
            En = E[l] / (gIL + gIU - 1)
            aUCEe[l] = aUCE ** e
            Ene[l] = En ** e

        # Lower spreading
        Es[self.Nc - 1] = Ene[self.Nc - 1]
        aLe = aL ** (-1 * e)
        for i in range((self.Nc - 2), -1, -1):
            Es[i] = aLe * Es[i + 1] + Ene[i]

        # Upper spreading (i > m)
        for i in range(0, (self.Nc - 1)):
            r = Ene[i]
            a = aUCEe[i]
            for l in range((i + 1), self.Nc):
                r = r * a
                Es[l] = Es[l] + r

        # Normalize the values by the normalization factor
        for i in range(0, self.Nc):
            Es[i] = (Es[i] ** (1 / e)) / Bs[i]

        return Es

    def PQ_timeSpread(self, Es, Ef):

        """
        电能质量时间扩散函数，用于对信号进行时间域的平滑处理
        参数:
            Es: 当前采样点的能量值
            Ef: 前一时刻的能量值
        返回:
            Ehs: 平滑后的最大能量值
            Ef: 更新后的前一时刻能量值
        """
        # 计算平滑因子Nadv，取NF的一半
        Nadv = self.NF // 2
        # 计算采样频率Fss
        Fss = float(self.Fs) / Nadv
        # 设置时间常数tau_100为0.030秒
        tau_100 = 0.030
        # 设置最小时间常数tau_min为0.008秒
        tau_min = 0.008
        # 计算alpha和beta系数，用于时间平滑
        alpha, beta = self.PQtConst(tau_100, tau_min, self.fc, Fss)

        # Allocate storage
        Ehs = np.zeros(self.Nc)
        # Time domain smoothing
        for i in range(self.Nc):
            Ef[i] = alpha[i] * Ef[i] + (1 - alpha[i]) * Es[i]
            Ehs[i] = max(Ef[i], Es[i])

        return Ehs, Ef

    def PQtConst(self, tau_100, tau_min, fc, Fss):
        """
        计算PQ控制器的常数alpha和beta
        参数:
            tau_100: 在100Hz时的tau值，单位为秒
            tau_min: 最小tau值，单位为秒
            fc: 截止频率数组
            Fss: 采样频率
        返回:
            alpha: 指数衰减系数
            beta: 补偿系数
        """
        # Tau values in units of seconds
        # tau_100 = 0.030
        # tau_min = 0.008

        # 初始化tau和alpha数组，长度与fc相同
        # tau = np.zeros(len(fc))
        # alpha = np.zeros(len(fc))

        # 计算tau值：根据截止频率fc线性插值计算
        tau = tau_min + (np.divide(float(100), fc)) * (tau_100 - tau_min)
        # 计算alpha值：基于时间常数tau和采样频率Fss的指数衰减
        alpha = np.exp(np.divide(-1. / Fss, tau))
        # 计算beta值：作为alpha的补偿系数
        beta = 1. - alpha

        return alpha, beta

    def PQIntNoise(self, f):
        """
        计算功率计内部噪声函数
        参数:
            f (float): 输入频率值，单位为Hz
        返回:
            float: 计算得到的等效输入噪声(EIN)
        """
        # 计算以dB为单位的内部噪声值
        # 使用经验公式：INdB = 1.456 * (f / 1000)^(-0.8)
        INdB = 1.456 * (f / 1000.) ** (-0.8)
        # 将dB值转换为线性值
        # 使用公式：EIN = 10^(INdB/10)
        EIN = 10 ** (INdB / 10.)
        return EIN

    def PQHannWin(self, NF):
        """
        汉宁窗函数生成器
        参数:
            NF (int): 窗函数的点数/长度
        返回:
            numpy.ndarray: 返回一个长度为NF的汉宁窗序列
        说明:
            汉宁窗是一种常用的窗函数，用于减少频谱泄漏。
            此函数使用numpy内置的hanning函数生成汉宁窗，
            等效于0.5*(1-cos(2*pi*n/(NF-1)))的公式计算。
        """
        # n = np.arange(0, NF)
        # hw = 0.5 * (1 - np.cos(2 * np.pi * n / (NF - 1)))
        # 直接使用numpy内置的hanning函数生成汉宁窗
        return np.hanning(NF)

    def PQRFFT(self, x, N, ifn):
        # Calculate the DFT of a real N-point sequence or the inverse
        # DFT corresponding to a real N-point sequence.
        # ifn > 0, forward transform
        #          input x(n)  - N real values
        #          output X(k) - The first N/2+1 points are the real
        #            parts of the transform, the next N/2-1 points
        #            are the imaginary parts of the transform. However
        #            the imaginary part for the first point and the
        #            middle point which are known to be zero are not
        #            stored.
        # ifn < 0, inverse transform
        #          input X(k) - The first N/2+1 points are the real
        #            parts of the transform, the next N/2-1 points
        #            are the imaginary parts of the transform. However
        #            the imaginary part for the first point and the
        #            middle point which are known to be zero are not
        #            stored. 
        #          output x(n) - N real values

        if (ifn > 0):
            X = np.fft.fft(x, N)
            XR = np.real(X[0:N // 2 + 1])
            XI = np.imag(X[1:N // 2 - 1 + 1])
            X = np.concatenate([XR, XI])
            return X
        else:
            raise Exception('ifft Not Implemented Yet -SW')

    def PQRFFTMSq(self, X, N):
        """
        Calculate the magnitude squared frequency response from the
        DFT values corresponding to a real signal (assumes N is even)
        """

        X2 = np.zeros(N // 2 + 1)
        X2[0] = X[0] ** 2
        for k in range(N // 2 - 1):
            X2[k + 1] = X[k + 1] ** 2 + X[N // 2 + k + 1] ** 2

        X2[N // 2] = X[N // 2] ** 2
        return X2

    def PQ_GL(self, NF=2048, Amax=1, fcN=1019.5 / 48000., Lp=92.):
        """
        Scaled Hann window, including loudness scaling
        Calculate the gain for the Hann Window
        level Lp (SPL) corresponds to a sine with normalized frequency
        fcN and a peak value of Amax
        """
        W = NF - 1
        gp = self.PQ_gp(fcN, NF, W)
        GL = 10 ** (Lp / 20.) / (gp * Amax / 4 * W)
        return GL

    def PQ_gp(self, fcN, NF, W):
        """
        Calculate the peak factor. The signal is a sinusoid windowed with
        a Hann window. The sinusoid frequency falls between DFT bins. The
        peak of the frequency response (on a continuous frequency scale) falls
        between DFT bins. The largest DFT bin value is the peak factor times
        the peak of the continuous response.
        fcN - Normalized sinusoid frequency (0-1)
        NF  - Frame (DFT) length samples
        NW  - Window length samples
        """

        # Distance to the nearest DFT bin
        df = 1. / NF
        k = np.floor(fcN / df)

        dfN = np.amin([(k + 1) * df - fcN, fcN - k * df])

        dfW = dfN * W
        gp = np.sin(np.pi * dfW) / (np.pi * dfW * (1 - dfW ** 2))
        return gp

    def PQWOME(self, f):
        """
        Generate the weighting for the outer & middle ear filtering
        Note: The output is a magnitude-squared vector
        """
        N = len(f)
        W2 = np.zeros(N)

        for k in range(N - 1):
            fkHz = float(f[k + 1]) / 1000
            AdB = -2.184 * fkHz ** (-0.8) + 6.5 * np.exp(-0.6 * (fkHz - 3.3) ** 2) - 0.001 * fkHz ** 3.6
            W2[k + 1] = 10 ** (AdB / 10)
        return W2

    def PQCB(self):
        # Critical band parameters for the FFT model, for Basic Version:
        dz = 1. / 4

        # I don't see why we can't hardcode this:
        Nc = 109

        fl = np.array([80.000, 103.445, 127.023, 150.762, 174.694,
                       198.849, 223.257, 247.950, 272.959, 298.317,
                       324.055, 350.207, 376.805, 403.884, 431.478,
                       459.622, 488.353, 517.707, 547.721, 578.434,
                       609.885, 642.114, 675.161, 709.071, 743.884,
                       779.647, 816.404, 854.203, 893.091, 933.119,
                       974.336, 1016.797, 1060.555, 1105.666, 1152.187,
                       1200.178, 1249.700, 1300.816, 1353.592, 1408.094,
                       1464.392, 1522.559, 1582.668, 1644.795, 1709.021,
                       1775.427, 1844.098, 1915.121, 1988.587, 2064.590,
                       2143.227, 2224.597, 2308.806, 2395.959, 2486.169,
                       2579.551, 2676.223, 2776.309, 2879.937, 2987.238,
                       3098.350, 3213.415, 3332.579, 3455.993, 3583.817,
                       3716.212, 3853.817, 3995.399, 4142.547, 4294.979,
                       4452.890, 4616.482, 4785.962, 4961.548, 5143.463,
                       5331.939, 5527.217, 5729.545, 5939.183, 6156.396,
                       6381.463, 6614.671, 6856.316, 7106.708, 7366.166,
                       7635.020, 7913.614, 8202.302, 8501.454, 8811.450,
                       9132.688, 9465.574, 9810.536, 10168.013, 10538.460,
                       10922.351, 11320.175, 11732.438, 12159.670, 12602.412,
                       13061.229, 13536.710, 14029.458, 14540.103, 15069.295,
                       15617.710, 16186.049, 16775.035, 17385.420])
        fc = np.array([91.708, 115.216, 138.870, 162.702, 186.742,
                       211.019, 235.566, 260.413, 285.593, 311.136,
                       337.077, 363.448, 390.282, 417.614, 445.479,
                       473.912, 502.950, 532.629, 562.988, 594.065,
                       625.899, 658.533, 692.006, 726.362, 761.644,
                       797.898, 835.170, 873.508, 912.959, 953.576,
                       995.408, 1038.511, 1082.938, 1128.746, 1175.995,
                       1224.744, 1275.055, 1326.992, 1380.623, 1436.014,
                       1493.237, 1552.366, 1613.474, 1676.641, 1741.946,
                       1809.474, 1879.310, 1951.543, 2026.266, 2103.573,
                       2183.564, 2266.340, 2352.008, 2440.675, 2532.456,
                       2627.468, 2725.832, 2827.672, 2933.120, 3042.309,
                       3155.379, 3272.475, 3393.745, 3519.344, 3649.432,
                       3784.176, 3923.748, 4068.324, 4218.090, 4373.237,
                       4533.963, 4700.473, 4872.978, 5051.700, 5236.866,
                       5428.712, 5627.484, 5833.434, 6046.825, 6267.931,
                       6497.031, 6734.420, 6980.399, 7235.284, 7499.397,
                       7773.077, 8056.673, 8350.547, 8655.072, 8970.639,
                       9297.648, 9636.520, 9987.683, 10351.586, 10728.695,
                       11119.490, 11524.470, 11944.149, 12379.066, 12829.775,
                       13294.850, 13780.887, 14282.503, 14802.338, 15341.057,
                       15899.345, 16477.914, 17077.504, 17690.045])
        fu = np.array([103.445, 127.023, 150.762, 174.694, 198.849,
                       223.257, 247.950, 272.959, 298.317, 324.055,
                       350.207, 376.805, 403.884, 431.478, 459.622,
                       488.353, 517.707, 547.721, 578.434, 609.885,
                       642.114, 675.161, 709.071, 743.884, 779.647,
                       816.404, 854.203, 893.091, 933.113, 974.336,
                       1016.797, 1060.555, 1105.666, 1152.187, 1200.178,
                       1249.700, 1300.816, 1353.592, 1408.094, 1464.392,
                       1522.559, 1582.668, 1644.795, 1709.021, 1775.427,
                       1844.098, 1915.121, 1988.587, 2064.590, 2143.227,
                       2224.597, 2308.806, 2395.959, 2486.169, 2579.551,
                       2676.223, 2776.309, 2879.937, 2987.238, 3098.350,
                       3213.415, 3332.579, 3455.993, 3583.817, 3716.212,
                       3853.348, 3995.399, 4142.547, 4294.979, 4452.890,
                       4643.482, 4785.962, 4961.548, 5143.463, 5331.939,
                       5527.217, 5729.545, 5939.183, 6156.396, 6381.463,
                       6614.671, 6856.316, 7106.708, 7366.166, 7635.020,
                       7913.614, 8202.302, 8501.454, 8811.450, 9132.688,
                       9465.574, 9810.536, 10168.013, 10538.460, 10922.351,
                       11320.175, 11732.438, 12159.670, 12602.412, 13061.229,
                       13536.710, 14029.458, 14540.103, 15069.295, 15617.710,
                       16186.049, 16775.035, 17385.420, 18000.000])

        return Nc, fc, fl, fu, dz

    def PQmodPatt(self):
        Nadv = self.NF // 2
        Fss = float(self.Fs) / Nadv
        tau_100 = 0.050
        tau_min = 0.008
        alpha, beta = self.PQtConst(tau_100, tau_min, self.fc, Fss)
        if self.check_PQmodPatt == False:
            self.DE = np.zeros((2, self.Nc))
            self.Ese = np.zeros((2, self.Nc))
            self.Eavg = np.zeros((2, self.Nc))
            self.check_PQmodPatt = True

        e = 0.3
        Ee = self.Es ** e
        alpha, beta = alpha[None], beta[None]
        self.DE = alpha * self.DE + beta * Fss * np.abs(Ee - self.Ese)
        self.Eavg = alpha * self.Eavg + beta * Ee
        self.Ese = Ee
        M = self.DE / (1 + self.Eavg / e)
        ERavg = self.Eavg[0]
        return M, ERavg

    def PQloud(self, Ehs, mod='FFT'):
        if mod != 'FFT':
            raise ValueError(f'Only FFT mod support, you choose {mod}')

        c = 1.07664
        e = 0.23
        E0 = 1e4
        self.Et = self.PQ_enThresh(self.fc)
        s = self.PQ_exIndex(self.fc)
        Ets = c * (self.Et / (s * E0)) ** e

        sN = np.sum(np.maximum(Ets * ((1 - s + s * Ehs / self.Et) ** e - 1), 0))
        Ntot = (24 / self.Nc) * sN
        return Ntot

    @staticmethod
    def PQ_enThresh(fc):
        """
        Calculate the energy threshold for each critical band.
        """
        return 10 ** ((3.64 * (fc / 1000) ** -0.8) / 10)

    @staticmethod
    def PQ_exIndex(fc):
        """
        计算PQ指数的函数
        参数:
            fc (float): 频率值，用于计算PQ指数
        返回:
            float: 通过特定公式计算得到的PQ指数值
        公式说明:
            该函数使用反正切函数(arctan)和幂运算来计算PQ指数
            公式为: 10的(((-2 - 2.05 * arctan(fc/4000) - 0.75 * arctan((fc/1600)^2)) / 10))次方
        """
        return 10 ** ((-2 - 2.05 * np.arctan(fc / 4000) - 0.75 * np.arctan((fc / 1600) ** 2)) / 10)

    def PQmovModDiffB(self, M, ERavg):

        """
        计算PQmovModDiffB函数，用于计算两个差异值和一个权重值

        参数:
            M: 输入矩阵，包含两个元素
            ERavg: 平均ER值，用于计算权重

        返回:
            三个计算结果:
            1. 第一个差异值，缩放后的s1B
            2. 第二个差异值，缩放后的s2B
            3. 计算得到的权重Wt
        """
        e = 0.3  # 指数参数，用于计算Ete
        Ete = self.EIN ** e  # 计算EIN的e次方
        negWt2B = 0.1  # 负权重因子
        offset1B = 1.0  # 第一个偏移量，用于避免除零
        offset2B = 0.01  # 第二个偏移量，用于避免除零
        levWt = 100  # 水平权重参数

        # 判断M[0]是否大于M[1]，用于后续条件计算
        cond = M[0] > M[1]
        # 根据条件计算num1B，确保结果为正数
        num1B = np.where(cond, M[0] - M[1], M[1] - M[0])
        # 根据条件计算num2B，可能引入负权重
        num2B = np.where(cond, negWt2B * num1B, num1B)
        # 计算第一个差异值MD1B，使用offset1B避免除零
        MD1B = num1B / (offset1B + M[0])
        # 计算第二个差异值MD2B，使用offset2B避免除零
        MD2B = num2B / (offset2B + M[0])
        # 计算MD1B的总和
        s1B = np.sum(MD1B)
        # 计算MD2B的总和
        s2B = np.sum(MD2B)
        # 计算权重Wt，考虑ERavg和levWt的影响
        Wt = np.sum(ERavg / (ERavg + levWt * Ete))

        # 返回缩放后的差异值和权重
        return (100 / self.Nc) * s1B, (100 / self.Nc) * s2B, Wt

    def PQmovPD(self, EhsR, EhsT):
        """
        计算PD_p和PD_q的值，用于评估信号质量
        参数:
            EhsR: 接收信号强度
            EhsT: 发射信号强度
        返回:
            PD_p: 概率密度函数值
            PD_q: 量化误差值
        """
        # 初始化多项式系数
        c = [-0.198719, 0.0550197, -0.00102438, 5.05622e-6, 9.01033e-11]
        # 初始化常数参数
        d1 = 5.95072
        d2 = 6.39468
        g = 1.71332
        bP = 4  # 正条件下的指数参数
        bM = 6  # 负条件下的指数参数

        # 将线性值转换为分贝值
        EdBR = 10 * np.log10(EhsR)
        EdBT = 10 * np.log10(EhsT)
        # 计算接收信号与发射信号的差值（分贝）
        edB = EdBR - EdBT

        # 判断信号差是否大于0
        cond = edB > 0
        # 根据条件选择L的计算方式
        L = np.where(cond, 0.3 * EdBR + 0.7 * EdBT, EdBT)
        # 根据条件选择b的值
        b = np.where(cond, bP, bM)

        # 判断L是否大于0
        cond = L > 0
        # 计算s值，使用多项式拟合
        s = np.where(cond, d1 * (d2 / L) ** g + c[0] + L * (c[1] + L * (c[2] + L * (c[3] + L * c[4]))), 1e30)

        # 计算PD_p值
        PD_p = 1 - 0.5 ** ((edB / s) ** b)
        # 计算PD_q值
        PD_q = np.abs(edB.astype(int)) / s
        return PD_p, PD_q


class PEAQ(object):
    def __init__(self, Amax=1, Fs=48000, NF=2048):

        """
        初始化函数，用于设置音频分析的基本参数
        参数:
            Amax (float): 最大信号振幅，默认值为1
            Fs (int): 采样频率，默认值为48000 Hz
            NF (int): 分析窗口的长度，默认值为2048
        """
        # Amax = maximum signal amplitude
        # Fs = sampling frequency
        # NF = Length of analysis window

        self.NF = NF  # 设置分析窗口的长度
        self.Fs = Fs  # 设置采样频率
        self.Amax = Amax  # 设置最大信号振幅

        # Step forward in half window lengths:
        self.Nadv = self.NF // 2  # 设置每次移动的步长为窗口长度的一半

        # Number of critical bands:
        self.Nc = 109  # 设置临界频带的数量为109
        self.P = np.zeros((2, self.Nc))  # 创建一个2x109的零矩阵，用于存储功率谱
        self.Rn = np.zeros((self.Nc))  # 创建一个长度为109的零向量，用于存储噪声功率
        self.Rd = np.zeros((self.Nc))  # 创建一个长度为109的零向量，用于存储失真功率
        self.PC = np.zeros((2, self.Nc))  # 创建一个2x109的零矩阵，用于存储感知功率谱

    def process(self, referenceSignal, testSignal):

        """
        处理参考信号和测试信号，执行音频质量评估
        参数:
            referenceSignal: 参考音频信号
            testSignal: 测试音频信号
        返回:
            无返回值，但会计算并存储多个评估指标
        """
        # 将输入信号赋值给局部变量，便于后续处理
        sigR = referenceSignal
        sigT = testSignal

        # Number of frames:
        self.Np = (np.floor(len(sigR) / self.Nadv)).astype(np.int32)

        # Scale audio:
        if np.amax(abs(sigR)) != self.Amax:
            # sigRS = self.Amax*sigR/float(np.amax(abs(sigR)))
            # sigTS = self.Amax*sigT/float(np.amax(abs(sigT)))
            sigRS = sigR
            sigTS = sigT
            print('Signals scaled, max reference value = ' + str(np.amax(abs(sigRS))) + ',')
            print('and max test value = ' + str(np.amax(abs(sigTS))) + '.')

        # Instantiate Object to process single frames of data:
        self.PQE = PQEval(Amax=self.Amax, Fs=self.Fs, NF=self.NF)

        print('Processing Audio...')

        # Create empty matrices:
        X2 = np.zeros((2, self.NF // 2 + 1))

        self.X2MatR = np.zeros((self.Np, self.NF // 2 + 1))
        self.X2MatT = np.zeros((self.Np, self.NF // 2 + 1))

        self.EbNMat = np.zeros((self.Np, self.Nc))
        self.EsMatR = np.zeros((self.Np, self.Nc))
        self.EsMatT = np.zeros((self.Np, self.Nc))

        self.EhsR = np.zeros((self.Np, self.Nc))
        self.EhsT = np.zeros((self.Np, self.Nc))

        previousFrameR = np.zeros(self.Nc)
        previousFrameT = np.zeros(self.Nc)

        # Maybe take this out later, but useful in debugging:
        self.xMatR = np.zeros((self.Np, self.NF))
        self.xMatT = np.zeros((self.Np, self.NF))

        self.loud_NRef = np.zeros((self.Np,))
        self.loud_NTest = np.zeros((self.Np,))

        self.BWRef = np.zeros((self.Np,))
        self.BWTest = np.zeros((self.Np,))

        self.PD_p = np.zeros((self.Np))
        self.PD_q = np.zeros((self.Np))
        self.MDiff_Mt1B = np.zeros((self.Np))
        self.MDiff_Mt2B = np.zeros((self.Np))
        self.MDiff_Wt = np.zeros((self.Np))
        self.NLoud_NL = np.zeros((self.Np))

        self.EHS = np.zeros((self.Np,))

        startS = 0

        startTime = time.time()

        for i in trange(self.Np):
            xR = sigRS[startS:self.NF + startS]
            xT = sigTS[startS:self.NF + startS]
            if xR.shape[-1] < self.NF:
                xR = np.pad(xR, (0, self.NF - xR.shape[-1]))
            if xT.shape[-1] < self.NF:
                xT = np.pad(xT, (0, self.NF - xT.shape[-1]))
            startS = startS + self.Nadv

            # Store unmodified windows of audio:
            self.xMatR[i, :] = xR
            self.xMatT[i, :] = xT

            # Process Frame:
            X2[0, :] = self.PQE.PQDFTFrame(xR)
            X2[1, :] = self.PQE.PQDFTFrame(xT)
            self.X2MatR[i, :] = X2[0, :]
            self.X2MatT[i, :] = X2[1, :]

            # Critical band grouping and frequency spreading
            self.EbN, self.Es = self.PQE.PQ_excitCB(X2)

            self.EbNMat[i, :] = self.EbN
            self.EsMatR[i, :] = self.Es[0, :]
            self.EsMatT[i, :] = self.Es[1, :]

            # Time domain spreading
            self.EhsR[i, :], previousFrameR = self.PQE.PQ_timeSpread(self.EsMatR[i, :], previousFrameR)
            self.EhsT[i, :], previousFrameT = self.PQE.PQ_timeSpread(self.EsMatT[i, :], previousFrameT)

            EP = self.PQadapt(self.EhsR[i], self.EhsT[i], 'FFT')
            M, ERavg = self.PQE.PQmodPatt()
            self.loud_NRef[i] = self.PQE.PQloud(self.EhsR[i, :])
            self.loud_NTest[i] = self.PQE.PQloud(self.EhsT[i, :])

            self.MDiff_Mt1B[i], self.MDiff_Mt2B[i], self.MDiff_Wt[i] = self.PQE.PQmovModDiffB(M, ERavg)

            self.NLoud_NL[i] = self.PQmovNLoudB(M, EP)

            self.BWRef[i], self.BWTest[i] = self.computeBW(self.X2MatR[i], self.X2MatT[i])

            PD_p, PD_q = self.PQE.PQmovPD(self.EhsR[i, :], self.EhsT[i, :])
            self.PD_p[i], self.PD_q[i] = self.PQ_ChanPD(PD_p, PD_q)

            self.EHS[i] = self.PQmovEHS(xR, xT, X2)

        ent_time = time.time()
        print('Processing complete. Time elapsed: ' + str(ent_time - startTime) + ' seconds.')
        self.NMRavg, self.NMRmax = self.computeNMR(self.EbNMat, self.EhsR)

    def PQ_ChanPD(self, p, q):
        """
        计算信道中断概率和信道容量
        参数:
            p: 每个子信道的中断概率列表
            q: 每个子信道的容量列表
        返回:
            Pc: 总的中断概率
            Qc: 总的信道容量
        """
        Pr = 1  # 初始化所有子信道都不中断的概率
        Qc = 0  # 初始化总信道容量
        for m in range(self.Nc):  # 遍历所有子信道
            Pr *= 1 - p[m]  # 计算所有子信道都不中断的概率
            Qc += q[m]  # 累加各子信道的容量
        Pc = 1 - Pr  # 计算至少一个子信道中断的概率
        return Pc, Qc  # 返回中断概率和总信道容量

    def get(self):
        """
        获取一个包含多个测量参数的字典

        返回:
            dict: 包含以下键的字典:
                - 'Ntot': 参考和测试的噪声计数
                    - 'NRef': 参考噪声计数
                    - 'NTest': 测试噪声计数
                - 'ModDiff': 模拟差值参数
                    - 'Mt1B': Mt1B模拟差值
                    - 'Mt2B': Mt2B模拟差值
                    - 'Wt': Wt模拟差值
                - 'NL': 响应噪声级别
                - 'BW': 带宽参数
                    - 'BWRef': 参考带宽
                    - 'BWTest': 测试带宽
                - 'NMR': 噪声测量比
                    - 'NMRavg': 平均噪声测量比
                    - 'NMRmax': 最大噪声测量比
                - 'PD': 概率分布参数
                    - 'p': p值
                    - 'q': q值
                - 'EHS': 等效听力级别
        """
        return {'Ntot': {'NRef': self.loud_NRef, 'NTest': self.loud_NTest},  # 返回总噪声计数及其参考和测试值
                'ModDiff': {'Mt1B': self.MDiff_Mt1B, 'Mt2B': self.MDiff_Mt2B, 'Wt': self.MDiff_Wt},  # 返回模拟差值参数
                'NL': self.NLoud_NL,  # 返回响应噪声级别
                'BW': {'BWRef': self.BWRef, 'BWTest': self.BWTest},  # 返回带宽参数及其参考和测试值
                'NMR': {'NMRavg': self.NMRavg, 'NMRmax': self.NMRmax},  # 返回噪声测量比及其平均值和最大值
                'PD': {'p': self.PD_p, 'q': self.PD_q},  # 返回概率分布参数
                'EHS': self.EHS}  # 返回等效听力级别

    def PQadapt(self, EhsR, EhsT, Mod='FFT'):
        """
        自适应处理函数，用于根据输入信号进行功率质量控制
        参数:
            EhsR: 右通道输入信号
            EhsT: 左通道输入信号
            Mod: 模式选择，目前仅支持'FFT'模式
        返回:
            EP: 处理后的双通道信号
        异常:
            ValueError: 当Mod参数不是'FFT'时抛出
        """
        if Mod != 'FFT':
            raise ValueError(f'Mod only supports FFT, but {Mod}')

        # 采样率相关参数设置
        Fs = 48000  # 采样率设为48kHz
        Fss = Fs / self.Nadv  # 计算子采样率
        t100 = 0.050  # 100ms时间常数
        tmin = 0.008  # 最小时间常数
        # 获取时间常数相关的系数
        a, b = self.PQE.PQtConst(t100, tmin, self.PQE.fc, Fss)
        M1, M2 = 3, 4  # 左右窗口大小参数

        # 初始化功率和比值的数组
        EP = np.zeros((2, self.Nc))  # 双通道误差功率
        R = np.zeros((2, self.Nc))  # 双通道比值

        # 更新功率估计
        self.P = np.expand_dims(a, -2) * self.P + np.expand_dims(b, -2) * np.stack([EhsR, EhsT])
        # 计算分子和分母
        sn = np.sum(np.sqrt(self.P[..., 0, :] * self.P[..., 1, :]), -1)
        sd = np.sum(self.P[..., 1, :], -1)

        # 计算控制信号
        CL = (sn / sd) ** 2
        cond = CL > 1
        EP[0] = np.where(cond, EhsR / CL, EhsR)
        EP[1] = np.where(cond, EhsT, EhsT * CL)

        # 更新归一化参数
        self.Rn = a * self.Rn + EP[1] * EP[0]
        self.Rd = a * self.Rd + EP[0] ** 2

        # 计算比值
        cond = self.Rn >= self.Rd
        R[0] = np.where(cond, 1, self.Rn / self.Rd)
        R[1] = np.where(cond, self.Rd / self.Rn, 1)

        for m in range(self.Nc):
            iL = max(m - M1, 0)
            iU = min(m + M2, self.Nc - 1)
            s1 = np.sum(R[0, iL:iU + 1], -1)
            s2 = np.sum(R[1, iL:iU + 1], -1)

            self.PC[0, m] = a[m] * self.PC[0, m] + b[m] * s1 / (iU - iL + 1)
            self.PC[1, m] = a[m] * self.PC[1, m] + b[m] * s2 / (iU - iL + 1)

            EP[0, m] *= self.PC[0, m]
            EP[1, m] *= self.PC[1, m]
        return EP

    def avg_get(self):
        """
        计算音频质量评估的多个指标平均值
        返回包含带宽、噪声调制差异、客观差异等级等指标的字典
        """
        # 计算带宽平均值
        self.avgBWRef, self.avgBWTest = self.PQ_avgBW(self.BWRef, self.BWTest)
        # 计算噪声调制比相关指标
        self.totalNMRB, self.relDistFramesB = self.PQ_avgNMRB(self.NMRavg, self.NMRmax)

        # 设置时间延迟参数
        tdel = 0.5
        # 计算采样频率和每帧样本数
        Fss = self.Fs / self.Nadv
        # 计算500毫秒对应的帧数
        N500ms = np.ceil(tdel * Fss)
        Nwup = 0
        # 计算延迟帧数，确保不小于0
        Ndel = np.maximum(np.zeros_like(N500ms), N500ms - Nwup)
        tex = 0.05

        # 计算调制差异相关指标
        self.WinModDiff1B, self.AvgModDiff1B, self.AvgModDiff2B = self.PQ_avgModDiffB(Ndel, self.MDiff_Mt1B,
                                                                                      self.MDiff_Mt2B, self.MDiff_Wt)
        # 计算相位差相关指标
        self.ADBB, self.MFPDB = self.PQ_avgPD(self.PD_p, self.PD_q)

        # 计算50毫秒对应的帧数
        N50ms = np.ceil(tex * Fss)
        # 计响度测试相关帧数
        Nloud = self.PQloudTest(self.loud_NRef, self.loud_NTest)
        # 更新延迟帧数，取较大值
        Ndel = max(Nloud + N50ms, Ndel)
        # 计算噪声响度相关指标
        self.RmsNoiseLoudB = self.PQ_avgNLoudB(Ndel, self.NLoud_NL)
        # 计算客观听力损伤指标
        self.EHSB = self.PQ_avgEHS(self.EHS)
        # 计算客观差异等级(ODG)
        self.ODG = self.PQnNetB(
            [self.avgBWRef, self.avgBWTest, self.totalNMRB, self.WinModDiff1B, self.ADBB, self.EHSB, self.AvgModDiff1B,
             self.AvgModDiff2B, self.RmsNoiseLoudB, self.MFPDB, self.relDistFramesB])
        # 返回包含所有计算指标的字典
        return {'BW': {'BWRef': self.avgBWRef, 'BWTest': self.avgBWTest},
                'NMR': {'totalNMRB': self.totalNMRB, 'relDistFramesB': self.relDistFramesB},
                'WinModDiff1B': self.WinModDiff1B,
                'AvgModDiff1B': self.AvgModDiff1B,
                'AvgModDiff2B': self.AvgModDiff2B,
                'ODG': self.ODG
                }

    def PQnNetB(self, MOV):
        """
        使用神经网络处理输入的MOV值并返回输出结果
        参数:
            MOV: 输入数据，将被转换为numpy数组进行处理
        返回:
            ODG: 神经网络处理后的输出值
        """
        # 获取神经网络的基本参数
        output = self.NNetPar('Basic')
        # 将输出参数转换为numpy数组
        amin, amax, wx, wxb, wy, wyb, bmin, bmax = list(map(np.array, output))
        # 将输入MOV转换为numpy数组
        MOV = np.array(MOV)
        # 获取权重矩阵wx的维度
        I, J = wx.shape

        # 对输入MOV进行归一化处理
        MOVx = (MOV - amin) / (amax - amin)
        # 初始化DI为 wyb
        DI = wyb
        # 遍历每一列
        for j in range(J):
            # 初始化arg为偏置wxb
            arg = wxb[j]
            # 遍历每一行
            for i in range(I):
                # 累加权重与归一化后的输入值的乘积
                arg += wx[i, j] * MOVx[i]
            # 累加激活函数的输出与对应权重的乘积
            DI += wy[j] * self.sigmoid(arg)
        # 计算最终输出结果
        ODG = bmin + (bmax - bmin) * self.sigmoid(DI)
        return ODG

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def NNetPar(Version):
        """
        根据版本返回神经网络参数
        参数:
            Version (str): 神经网络版本，'Basic'或其他
        返回:
            tuple: 包含以下元素的元组
                - amin: 最小激活值列表
                - amax: 最大激活值列表
                - wx: 输入层到隐藏层的权重矩阵
                - wxb: 隐藏层偏置
                - wy: 隐藏层到输出层的权重
                - wyb: 输出层偏置
                - bmin: 输出层最小值
                - bmax: 输出层最大值
        """
        if Version == 'Basic':
            # 基础版本的神经网络参数
            amin = [393.916656, 361.965332, -24.045116, 1.110661, -0.206623, 0.074318, 1.113683, 0.950345, 0.029985,
                    0.000101, 0]  # 最小激活值列表
            amax = [921, 881.131226, 16.212030, 107.137772, 2.886017,
                    13.933351, 63.257874, 1145.018555, 14.819740, 1,
                    1]  # 最大激活值列表
            wx = [[-0.502657, 0.436333, 1.219602],  # 输入层到隐藏层的权重矩阵
                  [4.307481, 3.246017, 1.123743],
                  [4.984241, -2.211189, -0.192096],
                  [0.051056, -1.762424, 4.331315],
                  [2.321580, 1.789971, -0.754560],
                  [-5.303901, -3.452257, -10.814982],
                  [2.730991, -6.111805, 1.519223],
                  [0.624950, -1.331523, -5.955151],
                  [3.102889, 0.871260, -5.922878],
                  [-1.051468, -0.939882, -0.142913],
                  [-1.804679, -0.503610, -0.620456]]
            wxb = [-2.518254, 0.654841, -2.207228]  # 隐藏层偏置
            wy = [-3.817048, 4.107138, 4.629582]  # 隐藏层到输出层的权重
            wyb = -0.307594  # 输出层偏置
            bmin = -3.98  # 输出层最小值
            bmax = 0.22  # 输出层最大值
        else:

            # 其他版本的神经网络参数
            amin = [13.298751, 0.041073, -25.018791, 0.061560, 0.024523]  # 最小激活值列表
            amax = [2166.5, 13.24326, 13.46708, 10.226771, 14.224874]  # 最大激活值列表
            wx = [[21.211773, -39.913052, -1.382553, -14.545348, -0.320899],  # 输入层到隐藏层的权重矩阵
                  [-8.981803, 19.956049, 0.935389, -1.686586, -3.238586],
                  [1.633830, -2.877505, -7.442935, 5.606502, -1.783120],
                  [6.103821, 19.587435, -0.240284, 1.088213, -0.511314],
                  [11.556344, 3.892028, 9.720441, -3.287205, -11.031250]]
            wxb = [1.330890, 2.686103, 2.096598, -1.327851, 3.087055]  # 隐藏层偏置
            wy = [-4.696996, -3.289959, 7.004782, 6.651897, 4.009144]  # 隐藏层到输出层的权重
            wyb = -1.360308  # 输出层偏置
            bmin = -3.98  # 输出层最小值
            bmax = 0.22  # 输出层最大值
        return amin, amax, wx, wxb, wy, wyb, bmin, bmax

    def PQ_avgEHS(self, EHS):
        """
        计算PQ（Power Quality）的平均EHS（电能质量指标）值
        参数:
            EHS: 输入的电能质量指标数据，可能是一个多维数组
        返回:
            返回一个缩放后的EHS平均值，乘以1000作为结果
        """
        # 计算EHS的线性位置平均值，并在最后一个维度上求和
        s = np.sum(self.PQ_LinPosAvg(EHS), -1)
        # 将求和结果乘以1000后返回
        return 1000 * s

    @staticmethod
    def PQ_LinPosAvg(x):

        """
        计算输入数组中非负数的平均值

        参数:
        x -- 输入数组或列表

        返回:
        非负数的算术平均值
        如果没有非负数，则返回NaN
        """
        return np.mean(x[x >= 0])  # 使用布尔索引筛选出非负数，然后计算平均值

    def PQloudTest(self, loud_NRef, loud_NTest):

        """
        PQloudTest函数用于执行某种测试，可能涉及参考信号和测试信号的比较。

        参数:
            loud_NRef: 参考信号的相关参数
            loud_NTest: 测试信号的相关参数

        返回:
            Ndel: 经过阈值处理后的结果值
        """
        Thr = 0.1  # 定义阈值常量Thr为0.1
        Ndel = self.Np  # 初始化Ndel为实例属性Np的值
        # 使用min函数比较当前Ndel和PQ_Lthresh函数返回值，取较小值作为新的Ndel
        Ndel = min(Ndel, self.PQ_Lthresh(Thr, loud_NRef, loud_NTest))
        return Ndel  # 返回处理后的Ndel值

    def PQ_Lthresh(self, Thr, loud_NRef, loud_NTest):

        """
        PQ_Lthresh函数用于查找两个信号中同时超过给定阈值的第一个点

        参数:
            Thr: 阈值，用于判断信号是否超过该值
            loud_NRef: 参考信号，包含多个采样点的值
            loud_NTest: 测试信号，包含多个采样点的值

        返回值:
            如果找到两个信号同时超过阈值的第一个点，则返回该点的索引
            如果没有找到这样的点，则返回总采样点数self.Np
        """
        for i in range(self.Np):  # 遍历所有采样点
            if loud_NRef[i] > Thr and loud_NTest[i] > Thr:  # 检查当前采样点是否两个信号都超过阈值
                return i  # 返回第一个满足条件的点索引
        return self.Np  # 如果没有找到满足条件的点，返回总采样点数

    def PQ_avgNLoudB(self, Ndel, NLoud):
        """
        计算平均噪声声压级
        参数:
            Ndel: int, 延迟点数，用于确定计算起点
            NLoud: numpy.ndarray, 噪声声压级数据数组
        返回:
            float, 计算得到的均方根噪声声压级，如果数据为空则返回0
        """
        # 从Ndel位置开始截取数据到数组末尾
        x = NLoud[int(Ndel):self.Np]
        # 检查截取后的数组是否为空
        if len(x) == 0:
            return 0
        # 计算均方根值（RMS），即平方和的平均值再开方
        return (np.sum(x ** 2, -1) / len(x)) ** 0.5

    def PQ_avgPD(self, PD_p, PD_q):

        """
        计算PD_p和PD_q的平均值和相关参数

        参数:
            PD_p: 输入参数P，用于计算Phc和Pcmax
            PD_q: 输入参数Q，用于计算ADBB

        返回:
            ADBB: 基于Qsum和nd计算的对数值或-0.5
            MFPDB: Pcmax的最大值
        """
        c0 = 0.9  # 用于计算Phc的系数
        c1 = 1  # 用于更新Pcmax的系数
        N = PD_p.shape[-1]  # 获取输入数组的最后一个维度大小
        nd = 0  # 计数器，用于统计PD_p大于0.5的元素数量
        Qsum = 0  # 用于累加PD_q中对应元素的值
        Pcmax = 0  # 用于记录Pcmax的最大值
        Phc = 0  # 用于计算加权平均值
        # 遍历数组中的每个元素
        for i in range(N):
            # 计算Phc的加权平均值
            Phc = c0 * Phc + (1 - c0) * PD_p[..., i]
            # 更新Pcmax，取当前值和Phc的较大值
            Pcmax = max(Pcmax * c1, Phc)
            # 检查当前PD_p值是否大于0.5
            if PD_p[i] > 0.5:
                nd += 1  # 增加计数器
                Qsum += PD_q[i]  # 累加对应的Q值

        # 根据条件计算ADBB的值
        if nd == 0:
            ADBB = 0  # 如果没有满足条件的元素，ADBB设为0
        elif Qsum > 0:
            ADBB = np.log10(Qsum / nd)  # 计算Qsum/nd的对数
        else:
            ADBB = -0.5  # 如果Qsum不大于0，ADBB设为-0.5

        MFPDB = Pcmax  # 将Pcmax的值赋给MFPDB
        return ADBB, MFPDB  # 返回计算结果

    def PQmovNLoudB(self, M, EP):

        """
        计算移动负载的噪声水平

        参数:
            M: 可能是质量或测量参数的数组
            EP: 可能是能量或功率参数的数组

        返回:
            计算得到的噪声水平值，如果小于最小值则返回0
        """
        alpha = 1.5  # 衰减系数
        TF0 = 0.15  # 初始时间因子
        S0 = 0.5  # 初始缩放因子
        NLmin = 0  # 最小噪声水平阈值
        e = 0.23  # 指数系数
        s = 0  # 初始化噪声水平变量

        # 计算参考噪声水平
        sref = TF0 * M[0] + S0
        # 计算测试噪声水平
        test = TF0 * M[1] + S0
        # 计算衰减因子beta
        beta = np.exp(-alpha * (EP[1] - EP[0]) / EP[0])
        # 计算临时变量tmp并确保非负
        tmp = test * EP[1] - sref * EP[0]
        a = np.maximum(tmp, np.zeros_like(tmp))
        # 计算分母部分b
        b = self.PQE.EIN + sref * EP[0] * beta
        # 计算噪声水平s
        s = np.sum((self.PQE.EIN / test) ** e * ((1 + a / b) ** e - 1))
        # 计算并归一化噪声水平NL
        NL = (24 / self.Nc) * s
        # 如果计算得到的噪声水平小于阈值，则返回0
        if NL < NLmin:
            return 0
        return NL

    def computeBW(self, X2MatR, X2MatT):
        """
        计算参考信号和测试信号的带宽

        参数:
            X2MatR: 参考信号的矩阵
            X2MatT: 测试信号的矩阵

        返回:
            BWRef: 参考信号的带宽值
            BWTest: 测试信号的带宽值
        """
        fx = 21586  # 中心频率
        kx = int(round(self.NF * float(fx) / self.Fs))  # 921
        fl = 8109
        kl = int(round(self.NF * float(fl) / self.Fs))  # 346
        FRdB = 10  # Ref. signal to exceed threshold level by 10dB
        FR = 10 ** (FRdB / 10.)
        FTdB = 5  # Test signal to exceed threshold level by 5dB
        FT = 10 ** (FTdB / 10.)

        Xth = np.amax(X2MatT[..., kx:-1], -1)
        XthR = FR * Xth
        cond = X2MatR[..., kl + 1:kx] >= XthR[..., None]
        BWRef = (np.arange(kl + 1, cond.shape[-1] + kl + 1)[None] * cond).max(-1) + 1
        # 确保 BWRef 是标量
        BWRef_scalar = BWRef.item() if np.isscalar(BWRef) else BWRef[0]

        XthT = FT * Xth
        cond = X2MatT[..., :int(BWRef_scalar - 1)] >= XthT[..., None]
        BWTest = (np.arange(cond.shape[-1])[None] * cond).max(-1) + 1

        BWTest_scalar = BWTest.item() if np.isscalar(BWTest) else BWTest[0]
        # 确保返回标量值
        return BWRef_scalar, BWTest_scalar

    def computeNMR(self, EbNMat, EhsR):
        """
        计算整个时间序列的归一化调制比(NMR)
        参数:
            EbNMat: 能量矩阵，包含每个时间点的能量值
            EhsR: 能量比率矩阵，包含每个时间点的能量比率
        返回:
            NMRavg: 平均归一化调制比数组
            NMRmax: 最大归一化调制比数组
        """
        # Kabal Section
        # Compute NRM for whole time series.

        NMRavg = np.zeros(self.Np)
        NMRmax = np.zeros(self.Np)

        for i in range(int(self.Np)):
            NMR = self.PQmovNMRB(EbNMat[i, :], EhsR[i, :])
            NMRavg[i] = NMR['NMRavg']
            NMRmax[i] = NMR['NMRmax']

        return NMRavg, NMRmax

    def PQmovNMRB(self, EbN, Ehs):

        """
        计算功率质量（PQ）中的噪声失真比（NMR）相关参数

        参数:
            EbN: 每个信道的信噪比列表
            Ehs: 每个信道的能量值列表

        返回:
            NMR: 包含最大噪声失真比和平均噪声失真比的字典
        """
        NMR = dict()  # 初始化一个空字典用于存储NMR相关结果

        # 获取功率质量相关参数
        Nc, fc, fl, fu, dz = self.PQE.PQCB()
        # 计算每个信道的掩码偏移量
        gm = self.PQ_MaskOffset(dz, Nc)

        NMRmax = 0  # 初始化最大噪声失真比为0
        NMRm = 0  # 初始化当前信道的噪声失真比为0
        s = 0  # 初始化噪声失真比总和为0

        # 初始化一个长度为Nc的零数组，用于存储每个信道的噪声失真比
        R_NM = np.zeros(Nc)

        # 遍历每个信道
        for k in range(Nc):
            # 计算当前信道的噪声失真比
            NMRm = EbN[k] / (gm[k] * Ehs[k])
            R_NM[k] = NMRm  # Remove later!
            s = s + NMRm

            if (NMRm > NMRmax):
                NMRmax = NMRm

        NMR['NMRmax'] = NMRmax
        NMR['NMRavg'] = float(s) / Nc

        return NMR

    def PQ_MaskOffset(self, dz, Nc):
        """
        计算PQ（Power Quality）掩码偏移量的函数
        参数:
            dz: float, 采样间隔或步长
            Nc: int, 采样点数或数组长度
        返回:
            gm: numpy.ndarray, 包含每个采样点对应的掩码偏移值
        """
        # 初始化一个长度为Nc的全零数组
        gm = np.zeros(Nc)
        # 遍历每个采样点
        for k in range(Nc):
            # 判断当前采样点是否在阈值范围内
            if k <= 12. / dz:
                # 在阈值范围内，设置mdB为3
                mdB = 3
            else:
                # 超出阈值范围，mdB随k和dz线性变化
                mdB = 0.25 * k * dz
            # 将mdB值转换为线性比例，并存储到gm数组中
            gm[k] = 10 ** (-1 * float(mdB) / 10)
        return gm

    def PQmovEHS(self, xR, xT, X2):
        """
        计算移动端语音质量评估的EHS(能量谐波失真)值

        参数:
            xR: 参考信号
            xT: 测试信号
            X2: 频谱相关系数

        返回:
            EHS: 能量谐波失真值，如果能量不足则返回-1
        """
        NF = 2048  # FFT点数
        Nadv = NF // 2  # 前向点数
        Fs = 48000  # 采样率
        Fmax = 9000  # 最大频率
        # 计算长度参数
        NL = 2 ** (self.PQ_log2(NF * Fmax / Fs))
        M = NL  # 窗口长度
        # 创建汉宁窗
        Hw = (1 / M) * (8 / 3) ** 0.5 * self.PQE.PQHannWin(M)

        EnThr = 8000  # 能量阈值
        kmax = NL + M - 1  # 最大索引

        # 复制并转换输入信号为float64类型
        xR, xT = np.copy(xR).astype(np.float64), np.copy(xT).astype(np.float64)

        # 计算参考信号和测试信号在特定区间的能量
        EnRef = np.matmul(xR[Nadv:NF + 1], xR[Nadv:NF + 1].T)
        EnTest = np.matmul(xT[Nadv:NF + 1], xT[Nadv:NF + 1].T)

        # 如果参考信号和测试信号的能量都低于阈值，返回-1
        if EnRef < EnThr and EnTest < EnThr:
            return -1

        # 计算对数谱差
        D = np.log(X2[1] / X2[0])
        # 计算相关系数
        C = self.PQ_Corr(D, NL, M)

        # 计算归一化相关系数
        Cn = self.PQ_NCorr(C, D, NL, M)
        # 计算归一化相关系数的平均值
        Cnm = (1 / NL) * np.sum(Cn[:NL.astype(int) + 1])

        # 应用汉宁窗并减去平均值
        Cw = Hw * (Cn - Cnm)

        # 进行实数FFT
        cp = self.PQE.PQRFFT(Cw, NL.astype(int), 1)
        # 计算幅度平方
        c2 = self.PQE.PQRFFTMSq(cp, NL.astype(int))

        # 查找峰值并返回EHS值
        EHS = self.PQ_FindPeak(c2, (NL / 2 + 1).astype(int))
        return EHS

    def PQ_Corr(self, D, NL, M):  # DFT-based operation in original matlab code

        """
        计算PQ相关性

        参数:
            D: 输入数据数组
            NL: 相关性计算的长度
            M: 数据的维度或大小

        返回:
            C: 计算得到的相关性数组
        """
        M = M.astype(int)  # 将M转换为整数类型
        NL = NL.astype(int)  # 将NL转换为整数类型

        C = np.zeros(NL)  # 初始化结果数组C，长度为NL
        for i in range(NL):  # 遍历NL次
            s = 0  # 初始化累加器s
            for j in range(M):  # 遍历M次
                s += D[..., j] * D[..., i + j]  # 计算D中元素的乘积并累加
            C[i] = s  # 将累加结果存入数组C
        return C  # 返回计算得到的相关性数组C

    @staticmethod
    def PQ_log2(x):

        """
        计算输入数组x以2为底的对数，向下取整
        参数:
            x: 输入数组或数值
        返回:
            与x形状相同的数组，每个元素是x对应元素以2为底的对数向下取整的结果
        """
        res = np.zeros_like(x)  # 创建与x形状相同，值全为0的数组作为结果
        m = 1  # 初始化乘数为1
        while m < x:  # 当乘数小于x时，继续循环
            res = res + 1  # 结果数组加1，相当于对数计数
            m *= 2  # 乘数乘以2，相当于2的幂次增加
        return res - 1  # 返回结果减1，因为循环多加了一次

    def PQ_NCorr(self, C, D, NL, M):
        """
        计算归一化相关系数
        """
        # 将输入参数NL和M转换为整数类型
        NL = NL.astype(int)
        M = M.astype(int)
        # 创建一个长度为NL的零数组，用于存储归一化后的相关系数
        Cn = np.zeros((NL,))

        # 获取C数组的第一个元素作为初始值s0
        s0 = C[0]
        # 初始化sj为s0，sj将用于累加D数组的平方差
        sj = s0
        # 设置归一化相关数组的第一个元素为1
        Cn[0] = 1
        # 循环计算从1到NL-1的归一化相关系数
        for i in range(1, NL):
            # 更新sj值，累加D数组中特定位置的平方差
            sj += (D[i + M - 1] ** 2 - D[i - 1] ** 2)
            # 计算分母d，即s0与sj的乘积
            d = s0 * sj
            # 如果分d小于等于0，则直接设置Cn[i]为1
            if d <= 0:
                Cn[i] = 1
            # 否则，计算归一化相关系数C[i]除以d的平方根
            else:
                Cn[i] = C[i] / d ** 0.5
        # 返回归一化后的相关系数数组
        return Cn

    @staticmethod
    def PQ_FindPeak(c2, N):

        """
        查找给定数组中的峰值元素
        峰值元素是指比其左右邻居都大的元素

        参数:
            c2: 包含数值的列表
            N: 列表的长度

        返回:
            cmax: 找到的峰值元素，如果没有峰值则返回0
        """
        cprev = c2[0]  # 保存前一个元素的值，初始化为第一个元素
        cmax = 0  # 用于记录找到的最大峰值，初始化为0
        for n in range(1, N):  # 从第二个元素开始遍历数组
            if c2[n] > cprev and c2[n] > cmax:  # 如果当前元素大于前一个元素且大于当前记录的最大值
                cmax = c2[n]  # 更新最大峰值
        return cmax  # 返回找到的最大峰值

    def PQ_avgBW(self, BWRef, BWTest):

        """
        计算参考带宽和测试带宽的平均值（仅计算非负值）
        参数:
            BWRef: 参考带宽数据，可以是数组或列表
            BWTest: 测试带宽数据，可以是数组或列表
        返回:
            tuple: 包含两个元素的元组
                - BandwidthRefB: 参考带宽的平均值（仅计算非负值）
                - BandwidthTestB: 测试带宽的平均值（仅计算非负值）
        """
        # 计算参考带宽中非负值的平均值
        BandwidthRefB = np.mean(BWRef[BWRef >= 0])
        # 计算测试带宽中非负值的平均值
        BandwidthTestB = np.mean(BWTest[BWTest >= 0])

        return BandwidthRefB, BandwidthTestB

    @staticmethod
    def PQ_avgNMRB(NMRavg, NMRmax):
        """
        计算平均NMRB值和相对距离帧比例
        参数:
        NMRavg: 平均NMR值数组
        NMRmax: 最大NMR值数组
        返回:
        totalNMRB: 平均NMRB值，单位为分贝
        relDistFramesB: 相对距离帧比例
        """
        # 计算平均NMRB值，使用对数变换
        totalNMRB = 10 * np.log10(np.mean(NMRavg))

        # Threshold:
        Tr = 10 ** (1.5 / 10)
        relDistFramesB = np.mean(NMRmax > Tr)

        return totalNMRB, relDistFramesB

    def PQ_avgModDiffB(self, Ndel, Mt1B, Mt2B, Wt):
        """
        计算调制差异的平均值，使用窗口加权和时间平均的方法

        参数:
            Ndel: 延迟样本数
            Mt1B: 第一个调制信号
            Mt2B: 第二个调制信号
            Wt: 加权窗口函数

        返回:
            WinModDiff1B: 窗口平均调制差异
            AvgModDiff1B: 第一个信号的时间加权平均调制差异
            AvgModDiff2B: 第二个信号的时间加权平均调制差异
        """
        NF = 2048  # FFT点数
        Nadv = NF / 2  # 每个advancement的样本数
        Fs = 48000  # 采样率(Hz)
        Ndel = int(Ndel)  # 将延迟样本数转换为整数

        Fss = Fs / Nadv  # 采样率除以advancement数，得到采样间隔
        tavg = 0.1  # 平均时间窗口(秒)

        L = np.floor(tavg * Fss)  # 计算平均窗口内的样本数
        # 计算第一个信号的窗口平均调制差异
        WinModDiff1B = self.PQ_WinAvg(int(L), Mt1B[Ndel:])

        # 计算两个信号的时间加权平均调制差异
        AvgModDiff1B = self.PQ_WtAvg(Mt1B[Ndel:], Wt[Ndel:])
        AvgModDiff2B = self.PQ_WtAvg(Mt2B[Ndel:], Wt[Ndel:])

        return WinModDiff1B, AvgModDiff1B, AvgModDiff2B

    @staticmethod
    def PQ_WinAvg(window_size, x):
        # N = len(x)
        #
        # s = 0
        # for i in range(L - 1, N):
        #     t = 0
        #     for m in range(L):
        #         t = t + np.sqrt(x[i - m])
        #     s = s + (t / L) ** 4
        # if N >= L:
        #     s = np.sqrt(s / (N - L + 1))
        # return s
        """
           Compute windowed average of input array using sliding window technique.

           For each position in the array, computes the 4th power of the average
           of square roots of values in a window of size L, then takes the square root
           of the mean of these values.

           Args:
               window_size (int): Size of the sliding window (L)
               x (np.ndarray): Input array to process

           Returns:
               float: Windowed average value

           Raises:
               ValueError: If window_size is not positive
               ValueError: If window_size is larger than input array
               ValueError: If input contains NaN or Inf values
               ValueError: If input contains negative values

           Note:
               - The function handles edge cases where window_size equals array length
               - Returns 0.0 for empty input or when window_size > len(x)
           """
        # Input validation
        if window_size <= 0:
            raise ValueError(f"Window size must be positive, got {window_size}")

        if len(x) == 0 or window_size > len(x):
            return 0.0

        # Check for invalid values
        if np.any(x < 0):
            raise ValueError("Input contains negative values (sqrt undefined)")

        # if np.any(~np.isfinite(x)):
        #     raise ValueError("Input contains NaN or Inf values")

        # Compute square roots once
        sqrt_x = np.sqrt(x)

        # Use sliding window with cumulative sum for efficiency
        # Compute cumulative sum of sqrt_x
        cumsum = np.cumsum(sqrt_x)

        # Initialize window sums array
        window_sums = np.zeros(len(x) - window_size + 1)

        # First window sum
        window_sums[0] = cumsum[window_size - 1]

        # Subsequent window sums using sliding window technique
        window_sums[1:] = cumsum[window_size:] - cumsum[:-window_size]

        # Compute window averages and raise to 4th power
        window_avgs = (window_sums / window_size) ** 4

        # Return square root of mean of window averages
        return np.sqrt(np.mean(window_avgs))

    @staticmethod
    def PQ_WtAvg(x, W):
        # N = len(x)
        # s = 0
        # sW = 0
        # for i in range(N):
        #     s = s + W[i] * x[i]
        #     sW = sW + W[i]
        #
        # if N > 0:
        #     s = s / sW
        # return s
        """
            Compute weighted average of input array.

            Args:
                x (np.ndarray): Input values to average
                W (np.ndarray): Weights for averaging (must be same length as x)

            Returns:
                float: Weighted average of x using weights W

            Raises:
                ValueError: If x and W have different lengths
                ValueError: If all weights are zero
                ValueError: If inputs contain NaN or Inf values

            Note:
                - Negative weights are allowed
                - The function handles the case where sum of weights is zero
            """
        # Input validation
        if len(x) != len(W):
            raise ValueError(f"Length mismatch: x has length {len(x)}, W has length {len(W)}")

        if len(x) == 0:
            return 0.0

        # # Check for NaN or Inf values
        # if np.any(~np.isfinite(x)) or np.any(~np.isfinite(W)):
        #     raise ValueError("Input contains NaN or Inf values")

        # Compute weighted average using NumPy operations
        weighted_sum = np.sum(W * x)
        weight_sum = np.sum(W)

        # Handle zero weight sum
        if weight_sum == 0:
            return 0.0

        return weighted_sum / weight_sum


def load(name):
    """
    加载音频文件并返回音频数据和采样率
    参数:
        name (str): 音频文件的路径
    返回:
        tuple: 包含两个元素的元组
            - audio (numpy.ndarray): 音频数据，转换为numpy数组
            - rate (int): 音频的采样率
    """
    # 使用torchaudio加载音频文件，不进行归一化处理
    audio, rate = torchaudio.load(name, normalize=False)
    # 如果音频数据类型是float32，将其转换为16位整数范围（-32768到32767）
    if audio.dtype == torch.float32:
        audio = audio * 32768.
    # 移除维度大小为1的维度，并将张量转换为numpy数组
    audio = audio.squeeze().numpy()
    return audio, rate


if __name__ == '__main__':
    ref, rate = load('../test_clean.wav')
    test, _ = load('../test_recons.wav')
    nppeaq = PEAQ(32768, Fs=rate)
    nppeaq.process(ref, test)
    metrics_as_frame = nppeaq.get()
    npmetrics = nppeaq.avg_get()
    print(npmetrics)
