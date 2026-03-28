import os
import tempfile
from typing import Dict, List
import gradio as gr
import pandas as pd
import numpy as np
import time
import librosa
import soundfile as sf
from audio_mos import ToneColorFidelityScore, DNSMOScore, NisqaMosScore, RefScore, WerScore, can_convert_to_float, \
    ScoreqScore
from audio_cut import cut_all_audio_files_from_list_multi_thread, cut_all_audio_files_from_list
from audio_align import align_splited_wav_from_list

# 计算模型初始化
ref_score_ins = RefScore()
dnsmos_calc_ins = DNSMOScore()
wer_calc_ins = WerScore()
nisqa_calc_ins = NisqaMosScore()
tcf_ins = ToneColorFidelityScore()
scoreq_ins = ScoreqScore()


def resample_audio_files(audio_files: List[str], target_sr: int = 16000) -> List[str]:
    """
    将上传的音频文件重采样到目标采样率
    :param audio_files: 原始音频文件路径列表
    :param target_sr: 目标采样率，默认16000Hz
    :return: 重采样后的音频文件路径列表
    """
    resampled_files = []

    for file_path in audio_files:
        try:
            # 读取音频文件
            y, sr = librosa.load(file_path, sr=None)

            # 如果采样率不同，则进行重采样
            if sr != target_sr:
                y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)

            # 保存重采样后的文件到临时目录
            temp_dir = tempfile.gettempdir()
            filename = os.path.basename(file_path)
            # name, ext = os.path.splitext(filename)
            # new_filename = f"{name}{ext}"
            new_file_path = os.path.join(temp_dir, filename)

            # 保存为wav格式
            sf.write(new_file_path, y, target_sr)
            resampled_files.append(new_file_path)

        except Exception as e:
            raise Exception(f"处理音频文件 {file_path} 时出错: {str(e)}")

    return resampled_files


def compute_mos_scores(audio_files: List[str], ref_dir="/mnt/test/scripts/speaker_verification/ref_dir/") -> Dict[
    str, List[float],]:
    """
    接收上传的音频路径列表，调用多个MOS计算器，并返回结构化结果。
    :param audio_files: list of uploaded file paths
    :param ref_dir: ref audio files path
    :return: dict like {"method_name": [score1, score2, ...]}
    """

    print("Processing audio files...")
    # print(audio_files)
    result = {}

    "stoi、sisdr、pesq"
    print("Calculating referenced scores...")
    try:
        ref_scores = ref_score_ins.get_mos(audio_files, ref_dir)
    except Exception as e:
        print("Error while calculating referenced mos scores...", e)
        return {}
    result.update(ref_scores)
    print("Calculating referenced scores successfully!!")

    "dnsmos"
    print("Calculating dnsmos scores...")
    try:
        dnsmos_scores = dnsmos_calc_ins.get_mos(audio_files)
    except Exception as e:
        print("Error while calculating dnsmos scores...", e)
        return {}
    result.update(dnsmos_scores)
    print("Calculating dnsmos scores successfully!!")

    "wer"
    print("Calculating wer based scores...")
    try:
        wer_scores = wer_calc_ins.get_wer(audio_files)
    except Exception as e:
        print("Error while calculating wer based mos scores...", e)
        return {}
    result.update(wer_scores)
    print("Calculating wer based scores successfully!!")

    "nisqa_dim"
    print("Calculating nisqa scores...")
    try:
        nisqa_dim_scores = nisqa_calc_ins.get_mos(audio_files)
    except Exception as e:
        print("Error while calculating nisqa scores...", e)
        return {}
    result.update(nisqa_dim_scores)
    print("Calculating nisqa scores successfully!!")

    "音色还原度"
    print("Calculating tone color fidelity scores...")
    try:
        tcf_scores = tcf_ins.get_mos(audio_files, ref_dir)
    except Exception as e:
        print("Error while calculating tone color fidelity scores...", e)
        return {}
    result.update(tcf_scores)
    print("Calculating tone color fidelity scores successfully!!")

    "scoreq得分"
    print("Calculating scoreq...")
    try:
        scoreq_scores = scoreq_ins.get_mos(audio_files)
    except Exception as e:
        print("Error while calculating scoreq...", e)
        return {}
    result.update(scoreq_scores)
    print("Calculating scoreq successfully!!")

    "整理成最终得分"
    values = np.array(list(result.values())).T
    final_scores = []
    for i in range(len(values)):
        val_tmp = [float(s) if can_convert_to_float(s) else s for s in values[i]]
        # print("val_tmp:",val_tmp)
        # print(len(val_tmp))
        # 加权 求最终分数
        tmp = np.mean([val_tmp[0] * 5, (1 / (1 + np.exp(-val_tmp[1]))) * 5, val_tmp[2],
                       val_tmp[3], val_tmp[4], val_tmp[6], (1 - val_tmp[7]) * 5, val_tmp[8] * 5,
                       val_tmp[9], val_tmp[10], val_tmp[11], val_tmp[12], val_tmp[13],
                       val_tmp[14] * 5, val_tmp[15]])
        final_scores.append(tmp)
    result.update({'final_scores': final_scores})
    return result


def toggle_button(disabled: bool):
    """切换按钮状态"""
    return gr.Button.update(interactive=not disabled)


def process_files(audio_files:List[str], progress=gr.Progress()):
    if not audio_files:
        yield None, "未上传任何音频文件", gr.update(interactive=True)
        return

    log_messages = ""

    def update_log(msg):
        nonlocal log_messages
        log_messages += f"\n[{time.strftime('%H:%M:%S')}] {msg}"
        # 返回: 文件路径, 日志消息, 按钮状态(计算中时禁用)
        yield None, log_messages, gr.update(interactive=False)

    yield from update_log("筛选有效音频文件...")
    valid_audio_files_list = []
    for file in audio_files:
        if file.endswith(".wav") or file.endswith(".mp3"):
            valid_audio_files_list.append(file)

    yield from update_log("开始处理音频文件...")

    try:
        # Step 1: Split audio files
        yield from update_log("正在切分音频...")
        split_file_list = cut_all_audio_files_from_list(
            valid_audio_files_list,
            output_dir="/mnt/test/scripts/label-studio-zch/label-studio-ml-backend/label_studio_ml/examples/voice_mos/split_out/",
            ref_dir="/mnt/test/scripts/label-studio-zch/label-studio-ml-backend/label_studio_ml/examples/voice_mos/ref_dir/"
        )

        # Step 2: Align audio segments
        yield from update_log("正在进行音频对齐...")
        aligned_file_list = align_splited_wav_from_list(
            input_file_list=split_file_list,
            ref_dir="/mnt/test/scripts/label-studio-zch/label-studio-ml-backend/label_studio_ml/examples/voice_mos/ref_dir/",
            output_dir="/mnt/test/scripts/label-studio-zch/label-studio-ml-backend/label_studio_ml/examples/voice_mos/est_dir/"
        )

        # Step 3: Compute MOS scores
        yield from update_log("正在计算各类 MOS 得分...")
        mos_results = compute_mos_scores(aligned_file_list)

        # Step 4: Generate DataFrame and save to Excel
        yield from update_log("生成结果表格...")
        df_data = {
            "文件名": [os.path.basename(file) for file in aligned_file_list],
        }
        for method, scores in mos_results.items():
            df_data[method] = scores

        df = pd.DataFrame(df_data)

        with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as tmpfile:
            df.to_excel(tmpfile.name, index=False)
            tmpfile_path = tmpfile.name

        final_message = log_messages + "\n✅ 所有任务已完成！请点击下方链接下载结果。"
        yield tmpfile_path, final_message, gr.update(interactive=True)

    except Exception as e:
        error_msg = f"❌ 发生错误: {str(e)}"
        final_message = log_messages + "\n" + error_msg
        yield None, final_message, gr.update(interactive=True)


# -----------------------------
# Gradio 接口定义
# -----------------------------
with gr.Blocks(title="MOS Score Calculator") as demo:
    gr.Markdown("# 🎵 MOS 分数计算工具")
    gr.Markdown("上传 WAV 格式音频文件，系统会自动将音频重采样到16KHz后计算MOS分数")

    # 添加示例音频下载区域
    gr.Markdown("## 📁 参考音频文件")
    with gr.Row():
        gr.Markdown("点击下方链接下载示例音频文件用于测试：")
    # 示例音频文件路径
    example_audio_path = "/mnt/test/scripts/label-studio-zch/label-studio-ml-backend/label_studio_ml/examples/voice_mos/speech_src.wav"  # � 替换为实际路径

    gr.Audio(
        label="示例音频试听",
        value=example_audio_path,
        type="filepath"
    )
    file_output = gr.Files(label="上传待分析音频文件 (支持多选)", file_types=[".wav",".mp3"])
    output_msg = gr.Textbox(label="状态信息", max_lines=20, interactive=False)
    download_btn = gr.File(label="下载结果 Excel 表格")
    calc_btn = gr.Button("开始计算", variant="primary")

    calc_btn.click(
        fn=process_files,
        inputs=file_output,
        outputs=[download_btn, output_msg, calc_btn],  # 添加按钮作为输出
        queue=True
    )
# -----------------------------
# 启动服务
# -----------------------------
if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=8003)
