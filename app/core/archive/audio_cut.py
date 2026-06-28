"""
【废弃】旧版MFCC定位切分模块

此模块已废弃, 由 matching_optimizer 替代:
  - MFCCLocate -> 全范围DTW扫描 (RobustDTWLocator + 39维MFCC+CMVN+余弦距离)
  - cut_all_audio_files_from_list -> cut_all_audio_files_with_optimized_matcher
     (当前函数已自动重定向到优化版，仅 import 失败时回退到旧实现)

保留作为回退, 新代码请使用 matching_optimizer。
"""

"""
根据1kHz标记切分音频文件
通过分析音频的短时傅里叶变换（STFT）来检测1kHz的频率峰值，然后根据这些峰值将音频切分成多个片段。

实际测试拿到手的音频录制文件，1k标记音都不太清晰，推荐手动将标记音放大后代替原来的标记位。
正常会切出4段音频，如果不是的话需要检查对应的文件。
"""
import heapq
import math
import os
import librosa
import numpy as np
import soundfile as sf
from scipy.signal import find_peaks

# 尝试导入dtw，如果失败则跳过
try:
    from dtw import dtw
    DTW_AVAILABLE = True
except ImportError:
    DTW_AVAILABLE = False
    print("警告: dtw模块未安装，部分功能可能不可用")

import matplotlib.pyplot as plt
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed


class FixedPriorityQueue:
    def __init__(self, max_size):
        self.max_size = max_size
        self.heap = []  # 最小堆 (优先级, 数据)

    def push(self, priority, item):
        """添加元素，保持队列不超过最大长度"""
        if len(self.heap) < self.max_size:
            heapq.heappush(self.heap, (priority, item))
        else:
            # 只保留优先级更高的元素（数值更小）
            if priority < self.heap[0][0]:
                heapq.heapreplace(self.heap, (priority, item))

    def pop(self):
        """弹出优先级最高的元素"""
        return heapq.heappop(self.heap) if self.heap else None

    def get_all(self):
        """获取所有元素（按优先级升序排序）"""
        return sorted(self.heap, key=lambda x: x[0])

    def __len__(self):
        return len(self.heap)

    def __str__(self):
        return str([f"({p}, {i})" for p, i in self.get_all()])

    def clear(self):
        while not self.empty():
            self.heap.pop()

    def empty(self):
        return len(self.heap) == 0


class MFCCLocate:
    def __init__(self, ref_file):
        self.ref_file = ref_file
        self.sr = 16000  # 统一使用16000采样率，与切分时一致
        self.hop_length = 512
        self.nfft = 2048
        # 按照这个配置，每一帧是间隔：512/16000=32ms 帧长：2048/16000=128ms
        self.n_mfcc = 13
        self.ref_mfcc, self.ref_y = self.extract_mfcc(self.ref_file, sr=self.sr, n_mfcc=self.n_mfcc)
        self.ref_time_length = len(self.ref_y) / self.sr
        print(f"MFCCLocate初始化: 参考文件={ref_file}, 采样率={self.sr}, 时长={self.ref_time_length:.3f}s")

    @staticmethod
    def extract_mfcc(audio_path, sr=16000, n_mfcc=13):
        """
        提取mfcc特征，调用librosa的方法
        Args:
            audio_path:
            sr:
            n_mfcc:
        Returns:

        """
        y, sr = librosa.load(audio_path, sr=sr)
        # nfft = 2048, hop_length = 512
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, )
        return mfcc.T, y  # 转置为(时间帧, 特征维度)
        # return mfcc.mean(axis=1).flatten()

    def index2time(self, index):
        """
        index ——> time stamp
        Args:
            index:

        Returns:

        """
        _time = index * (self.hop_length / self.sr)
        return _time

    def time2index(self, _time):
        """
        time stamp ——> index
        Args:
            _time:

        Returns:
        """
        _index = math.floor(_time * self.sr / self.hop_length)
        return _index

    def audio_locate(self, long_audio_path):
        """

        """
        # print(f"Searching in {long_audio_path} using reference file: {self.ref_file}")

        riddle_start_time = self._locate_short_audio_with_dtw(long_audio_path,
                                                              search_start_time=0,
                                                              search_stop_time=1e4,
                                                              jump_step=15)
        # 精筛，在粗筛结果的基础上，前后1s以内
        # 用更大的n_mfcc和更小的step
        # fine_start_time,fine_end_time = self._locate_short_audio_with_dtw(long_audio_path,
        #                                                     search_start_time=riddle_start_time - 0.5,
        #                                                     search_stop_time=riddle_start_time + 0.5,
        #                                                     jump_step=1)
        # print(f"fine_start_time:{fine_start_time}")
        # return fine_start_time,fine_end_time
        # 只需要粗筛+一定的冗余即可
        return riddle_start_time, self.ref_time_length

    def _locate_short_audio_with_dtw(self, long_audio_path, search_start_time, search_stop_time,
                                     jump_step):
        """
        筛选
        Args:
            long_audio_path:
            search_start_time:
            search_stop_time:
            jump_step:

        Returns:
        """
        if not DTW_AVAILABLE:
            raise ImportError("dtw模块未安装，无法使用DTW功能。请运行: pip install dtw-python")
        
        pq = FixedPriorityQueue(10)  # 最大长度

        # 提取特征
        long_mfcc, y = self.extract_mfcc(long_audio_path, sr=self.sr, n_mfcc=self.n_mfcc)
        # 滑动窗口对比
        # min_distance = float('inf')
        window_size = self.ref_mfcc.shape[0]
        # best_start_index = -1

        start_index = max(0, self.time2index(search_start_time))
        stop_index = min(len(long_mfcc) - window_size + 1, self.time2index(search_stop_time))

        for i in range(start_index, stop_index, jump_step):
            window = long_mfcc[i:i + window_size]

            # dtw法
            distance = dtw(window, self.ref_mfcc, dist_method='euclidean')
            pq.push(priority=distance.distance, item=i)

        if not pq.empty():
            _, best_start_index = pq.pop()
        else:
            best_start_index = 0

        start_time = self.index2time(best_start_index)
        return start_time


class CutUsing1k:
    def __init__(self):
        pass

    @staticmethod
    def _find_longest_path(queue) -> list:
        """
        在一个队列中找到一条最长路径，使得每个点的差值都在12.5~14.5之间
        """
        n = len(queue)
        longest_path = []

        def dfs(index, current_path):
            nonlocal longest_path
            # 更新最长路径
            if len(current_path) > len(longest_path):
                longest_path = current_path[:]

            # 尝试扩展路径
            for i in range(index + 1, n):
                diff = abs(queue[i] - queue[index])
                if 12.5 <= diff <= 14.5:
                    current_path.append(queue[i])
                    dfs(i, current_path)
                    current_path.pop()

        # 以每个元素为起点进行深度优先搜索
        for i in range(n):
            dfs(i, [queue[i]])

        return longest_path

    def _check_peaks(self, peak_list, target_peaks_num=4):
        """
        检查1k峰值搜索结果
        预期峰值队列是以13~14s为间隔
        """
        if len(peak_list) < target_peaks_num:
            print("Peak number is less than 4, please check the audio file.")
            return None

        return self._find_longest_path(peak_list)

    def split_audio_by_1khz_markers(self, input_path, output_prefix, freq_target=1000, duration=0.3, threshold_db=-30,
                                    default_sr=16000):
        """
        根据1kHz标记切分音频文件
        实现原理
        加载音频: 使用librosa.load加载音频文件，并将其转换为单声道。
        计算STFT: 使用librosa.stft计算音频的短时傅里叶变换，得到频率和时间的二维数组。
        提取1kHz频段: 找到1kHz对应的频率索引，并提取该频段的能量。
        能量转换: 将能量转换为分贝，以便于后续的峰值检测。
        峰值检测: 使用scipy.signal.find_peaks检测能量中的峰值，这些峰值对应于1kHz的标记信号。
        切分音频: 根据检测到的峰值位置，将音频切分成多个片段，并保存为新的音频文件。

        注意阈值设的比较极端，配合手动修改1k标记音，这样切出来比较准
        参数:
            input_path: 输入音频路径
            output_prefix: 输出片段前缀（如output_dir/split_）
            freq_target: 目标频率（Hz），默认为1kHz
            duration: 标记信号时长（秒），默认为0.5s
            threshold_db: 能量阈值（dB），低于此值忽略
            dafault_sr: 音频采样率，默认为44100Hz
        """
        print(f"Splitting {input_path} by 1kHz markers...")
        # 需要加一些偏移量，保证切出来的音频里头没有1k
        left_shift = 1
        right_shift = 13
        # 加载音频
        y, sr = librosa.load(input_path, sr=default_sr, mono=True)

        # 计算STFT
        n_fft = 2048
        hop_length = 512
        stft = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length))

        # 提取1kHz频段
        freq_bin = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
        target_idx = np.argmin(np.abs(freq_bin - freq_target))
        energy = stft[target_idx, :]

        # 转换为分贝并归一化
        energy_db = librosa.amplitude_to_db(energy, ref=np.max)

        # 画图显示
        # plt.plot(energy_db)
        # plt.show()

        # 检测能量峰值（对应标记信号）
        min_samples = int(duration * sr / hop_length)  # 最小持续点数
        # 调用scipy的峰值搜索函数，通过阈值和持续时间进行筛选
        peaks, _ = find_peaks(energy_db, height=threshold_db, width=min_samples)

        # 转换为时间戳（秒）
        times = librosa.times_like(energy_db, sr=sr, hop_length=hop_length)
        split_points = times[peaks]

        print("Possible splits: ", split_points)
        # 对1k峰值搜索结果进行检查
        split_points = self._check_peaks(split_points)
        if split_points is None:
            raise RuntimeWarning("1k峰值搜索结果异常，请检查音频文件")

        print("Find 1k in: ", split_points)

        # 切分并保存音频片段
        last_end_sample = 0
        generated_index = 11
        for i in range(len(split_points)):
            start_sample = int((left_shift + split_points[i]) * sr)
            end_sample = int((right_shift + split_points[i]) * sr)
            if start_sample <= last_end_sample:
                # 开始应该是在下一次的后面
                print(f"音频段重叠，跳过音频段 {start_sample / sr}s to {end_sample / sr}s")
                continue
            last_end_sample = end_sample
            # end_sample应该要小于序列总长，否则就说明后面这段并没有人声或者人声不完整
            if end_sample - 2 * sr > len(y):
                print(f"音频长度不足，跳过音频段 {start_sample / sr}s to {len(y) / sr}s")
                continue
            segment = y[start_sample:end_sample]
            sf.write(f"{output_prefix}_{generated_index:03d}.wav", segment, sr)
            print(f"保存音频段 {start_sample / sr}s to {end_sample / sr}s to {output_prefix}_{generated_index:03d}.wav")
            generated_index += 1
        if generated_index != 5:
            raise RuntimeWarning("切分音频段数量不正确，请检查音频文件")


def cut_all_audio_files_in_directory_1k(input_dir, output_dir):
    """
    处理目录中的所有WAV文件
    参数:
        input_dir: 输入目录
        output_dir: 输出目录
    """
    cu1k = CutUsing1k()
    for filename in os.listdir(input_dir):
        if filename.endswith(".wav"):
            input_path = os.path.join(input_dir, filename)
            output_prefix = os.path.join(output_dir, os.path.splitext(filename)[0])
            cu1k.split_audio_by_1khz_markers(input_path, output_prefix)


def cut_all_audio_files_from_list(input_file_list, ref_dir, output_dir):
    """
    输入音频文件列表，返回处理后文件保存的路径

    注意: 此函数已废弃，由 matching_optimizer.cut_all_audio_files_with_optimized_matcher
          替代（全范围DTW+HPSS精对齐，低SNR下更可靠）。
          当前实现优先尝试加载优化版，失败时回退到旧版MFCCLocate。
    """
    # 优先使用优化版（全范围DTW+HPSS精对齐）
    _opt_warned = False
    try:
        from matching_optimizer import cut_all_audio_files_with_optimized_matcher
        logger = logging.getLogger('audiomos')
        logger.warning("[废弃] audio_cut.cut_all_audio_files_from_list 被调用，"
                       "建议使用 matching_optimizer.cut_all_audio_files_with_optimized_matcher")
        return cut_all_audio_files_with_optimized_matcher(input_file_list, ref_dir, output_dir)
    except ImportError:
        pass
    except Exception as e:
        if not _opt_warned:
            logging.getLogger('audiomos').warning(f"优化切分失败,回退到旧版MFCCLocate: {e}")
            _opt_warned = True

    # 旧版MFCCLocate实现（回退）
    ref_files = [os.path.join(ref_dir, f) for f in os.listdir(ref_dir) if f.endswith(".wav")]
    # 要排序
    ref_files.sort()
    print(f"【切分算法】找到{len(ref_files)}个参考音频: {[os.path.basename(f) for f in ref_files]}")
    ml_models = [MFCCLocate(ref_file=rf) for rf in ref_files]

    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    # 处理每一个输入文件
    output_file_list = []
    for file in input_file_list:
        print(f"\n【切分算法】处理文件: {file}")

        for idx, ml in enumerate(ml_models):
            output_path = None
            # 执行音频定位
            try:
                found_start_time, duration = ml.audio_locate(file)
                print(f"【切分算法】DTW定位结果: 参考音频{idx+1}在长音频中的起始时间={found_start_time:.3f}s, 参考音频时长={duration:.3f}s")

                # 加载对应的片段（前后各加0.5秒冗余）
                # 注意：
                # 1. 参考音频本身是从0s开始的
                # 2. found_start_time是参考音频在长音频中的起始位置
                # 3. 为了保留前后冗余，我们使用offset = found_start_time - 0.5
                # 4. 这样切分后的音频中，目标内容从0.5s开始（与参考音频的0s对齐）
                actual_offset = max(0.0, found_start_time - 0.5)
                y, sr = librosa.load(
                    file,
                    sr=16000,
                    offset=actual_offset,
                    duration=duration + 0.5
                )
                print(f"【切分算法】切分参数: offset={actual_offset:.3f}s, duration={duration + 0.5:.3f}s")
                print(f"【切分算法】对齐说明: 切分音频中，目标内容从{found_start_time - actual_offset:.3f}s开始，与参考音频的0s对齐")

                # 构造输出文件名
                base_name = os.path.splitext(os.path.basename(file))[0]
                suffix = f"_00{idx + 1}.wav"
                output_file_name = base_name + suffix
                output_path = os.path.join(output_dir, output_file_name)

                # 写入音频文件
                sf.write(output_path, y, sr)
                print(f"【切分算法】已保存: {output_file_name}")
            except Exception as e:
                print(f"【切分算法】错误: {e}")
            if output_path is not None:
                output_file_list.append(output_path)
    return output_file_list


# 线程锁用于保护共享资源
output_lock = threading.Lock()


def process_single_file(file, ml_models, output_dir, output_file_list):
    """处理单个音频文件"""
    print("Processing:", file)
    local_output_files = []

    for idx, ml in enumerate(ml_models):
        try:
            # 执行音频定位
            found_start_time, duration = ml.audio_locate(file)

            print(file, idx + 1, found_start_time, duration)
            # 加载对应的片段（前后各加0.5秒）
            actual_offset = max(0.0, found_start_time - 0.5)
            y, sr = librosa.load(
                file,
                sr=16000,
                offset=actual_offset,
                duration=duration + 1
            )
            print(f"切分信息: 找到的起始时间={found_start_time:.3f}s, 实际offset={actual_offset:.3f}s, 持续时间={duration:.3f}s")
            # 构造输出文件名
            base_name = os.path.splitext(os.path.basename(file))[0]
            suffix = f"_00{idx + 1}.wav"
            output_file_name = base_name + suffix
            output_path = os.path.join(output_dir, output_file_name)
            # 写入音频文件
            sf.write(output_path, y, sr)
            print("Split file and stored to", output_file_name)
            local_output_files.append(output_path)

        except Exception as e:
            print(f"Error processing {file} with model {idx}: {e}")
            continue

    # 线程安全地添加到全局结果列表
    with output_lock:
        output_file_list.extend(local_output_files)


def cut_all_audio_files_from_list_multi_thread(input_file_list, ref_dir, output_dir, max_workers=4):
    """
    输入音频文件列表，返回处理后文件保存的路径（多线程版本）
    input_file_list:输入文件列表
    ref_dir:参考文件路径
    output_dir:输出文件路径
    max_workers:最大线程数
    """
    ref_files = [os.path.join(ref_dir, f) for f in os.listdir(ref_dir) if f.endswith(".wav")]
    # 要排序
    ref_files.sort()
    ml_models = [MFCCLocate(ref_file=rf) for rf in ref_files]

    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    # 处理所有输入文件（多线程）
    output_file_list = []

    # 使用ThreadPoolExecutor管理线程
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 提交任务
        future_to_file = {
            executor.submit(process_single_file, file, ml_models, output_dir, output_file_list): file
            for file in input_file_list
        }

        # 等待所有任务完成
        for future in as_completed(future_to_file):
            file = future_to_file[future]
            try:
                future.result()  # 获取结果（如果有异常会在这里抛出）
            except Exception as e:
                print(f"Error processing {file}: {e}")
    output_file_list.sort()
    return output_file_list


def cut_all_audio_files_in_directory_dtw(input_dir, ref_dir, output_dir, max_workers=4):
    """处理目录中所有wav文件（多线程版本）"""
    inpit_file_list = []
    for filename in os.listdir(input_dir):
        if filename.endswith(".wav"):
            input_path = os.path.join(input_dir, filename)
            inpit_file_list.append(input_path)

    return cut_all_audio_files_from_list_multi_thread(inpit_file_list, ref_dir, output_dir, max_workers)
    # return cut_all_audio_files_from_list(inpit_file_list, ref_dir, output_dir)


# ============================================================================
# 基于内容匹配的切分方法（使用ReferencePipeline，支持动态参考音频集合）
# ============================================================================

def cut_all_audio_files_with_content_matching(
    input_file_list: list,
    ref_dir: str,
    output_dir: str,
    use_dtw: bool = True,
    min_confidence: float = 0.3
) -> list:
    """
    使用内容匹配（指纹+DTW）切分音频文件
    这是音频切分的新主入口，支持任意数量的自定义参考音频。

    工作流程：
    1. 建立参考音频指纹数据库
    2. 对每个测试音频进行内容匹配
    3. 切分并对齐匹配到的片段
    4. 返回对齐后的音频文件路径列表

    Args:
        input_file_list: 输入音频文件路径列表
        ref_dir: 参考音频目录
        output_dir: 输出目录
        use_dtw: 是否使用DTW精确定位
        min_confidence: 最低匹配置信度

    Returns:
        对齐后的音频文件路径列表
    """
    # 延迟导入，避免循环依赖
    try:
        from reference_pipeline import ReferencePipeline, process_multiple_test_files
    except ImportError:
        # 如果从其他位置调用，尝试从app.core导入
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
        from reference_pipeline import ReferencePipeline, process_multiple_test_files

    import logging
    logger = logging.getLogger('audiomos')

    logger.info(f"[内容匹配切分] 开始处理 {len(input_file_list)} 个文件")
    logger.info(f"[内容匹配切分] 参考目录: {ref_dir}")
    logger.info(f"[内容匹配切分] 输出目录: {output_dir}")

    os.makedirs(output_dir, exist_ok=True)

    pipeline = ReferencePipeline(ref_dir=ref_dir)
    pipeline.initialize(ref_dir, force_rebuild=True)

    all_aligned_files = []

    for test_path in input_file_list:
        result = pipeline.process_test_audio(
            test_audio_path=test_path,
            output_dir=output_dir,
            min_confidence=min_confidence,
            use_dtw=use_dtw
        )

        if result["no_match"]:
            logger.warning(f"[内容匹配切分] ⚠️ 未找到匹配的参考音频: {os.path.basename(test_path)}")
        else:
            aligned = result["aligned_files"]
            logger.info(f"[内容匹配切分] ✓ {os.path.basename(test_path)}: "
                         f"{len(result['matches'])} 个匹配, {len(aligned)} 个对齐文件")
            all_aligned_files.extend(aligned)

    logger.info(f"[内容匹配切分] 处理完成: 共 {len(all_aligned_files)} 个对齐文件")
    return all_aligned_files


# 示例调用
if __name__ == "__main__":
    # cut_all_audio_files_in_directory_1k(input_dir=r"E:\阵列麦克风提测数据\251106试产后软件第一轮",
    #                                    output_dir="split_out")
    # cut_all_audio_files_in_directory_dtw(
    #     input_dir="/mnt/test/scripts/label-studio-zch/label-studio-ml-backend/label_studio_ml/examples/voice_mos/test_audio/",
    #     output_dir="/mnt/test/scripts/label-studio-zch/label-studio-ml-backend/label_studio_ml/examples/voice_mos/split_out/",
    #     ref_dir="/mnt/test/scripts/label-studio-zch/label-studio-ml-backend/label_studio_ml/examples/voice_mos/ref_dir/",
    #     max_workers=8)
    print(cut_all_audio_files_in_directory_dtw(
        input_dir=r"E:\阵列麦克风提测数据\251106试产后软件第一轮",
        output_dir="split_out",
        ref_dir="ref_dir",
        max_workers=8))
