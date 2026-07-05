"""
批量音频修复性能测试脚本
测试批量处理vs单文件处理的性能差异
"""
import os
import time
import requests
import glob
from datetime import datetime

# ── 配置 ──
BASE_URL = "http://127.0.0.1:8002/api"
AUDIO_DIR = "/home/zhouchenghao/桌面/拾音距离/拾音距离"
USERNAME = "admin"
PASSWORD = "tp123456"
ALGORITHM = "dereverberation"  # 使用去混响算法测试
TEST_FILES_COUNT = 40  # 测试所有40个文件

# ── 全局变量 ──
token = None
results = {}


def login():
    """登录获取token"""
    global token  # 使用全局变量
    
    print("\n" + "="*60)
    print("  步骤1: 登录系统")
    print("="*60 + "\n")
    
    login_url = f"{BASE_URL}/auth/login"
    response = requests.post(
        login_url,
        data={"username": USERNAME, "password": PASSWORD}
    )
    
    if response.status_code == 200:
        token = response.json()["access_token"]
        print(f"✅ 登录成功 - 用户: {USERNAME}")
        print(f"   Token: {token[:20]}...")
        return token
    else:
        print(f"❌ 登录失败 - {response.text}")
        raise Exception("登录失败")


def get_audio_files(count=TEST_FILES_COUNT):
    """获取测试音频文件列表"""
    print("\n" + "="*60)
    print("  步骤2: 准备测试音频文件")
    print("="*60 + "\n")
    
    audio_files = glob.glob(os.path.join(AUDIO_DIR, "*.wav"))
    audio_files = sorted(audio_files)[:count]
    
    print(f"✅ 找到 {len(audio_files)} 个音频文件")
    
    # 显示前5个文件信息
    print("\n   前5个文件:")
    for i, file_path in enumerate(audio_files[:5], 1):
        filename = os.path.basename(file_path)
        file_size = os.path.getsize(file_path) / 1024 / 1024
        print(f"      {i}. {filename} ({file_size:.2f}MB)")
    
    print(f"\n   总大小: {sum(os.path.getsize(f) for f in audio_files) / 1024 / 1024:.2f}MB")
    return audio_files


def test_single_file_processing(audio_files):
    """测试单文件串行处理(对照组)"""
    print("\n" + "="*60)
    print("  步骤3: 单文件串行处理测试(对照组)")
    print("="*60 + "\n")
    
    # 只测试前10个文件作为对照组
    test_files = audio_files[:10]
    total_time = 0
    
    print(f"测试文件数量: {len(test_files)}")
    print(f"算法: {ALGORITHM}")
    print("\n开始串行处理...\n")
    
    start_time = time.time()
    
    for i, file_path in enumerate(test_files, 1):
        filename = os.path.basename(file_path)
        print(f"[{i}/{len(test_files)}] 处理: {filename}")
        
        # 1. 上传文件
        upload_start = time.time()
        upload_url = f"{BASE_URL}/restoration/upload"
        
        with open(file_path, "rb") as f:
            response = requests.post(
                upload_url,
                headers={"Authorization": f"Bearer {token}"},
                files={"file": (filename, f)},
                data={"algorithm": ALGORITHM}
            )
        
        upload_time = time.time() - upload_start
        
        if response.status_code != 200:
            print(f"  ❌ 上传失败: {response.text}")
            continue
        
        task_id = response.json()["task_id"]
        print(f"  ✓ 上传成功 (耗时: {upload_time:.2f}s)")
        
        # 2. 提交处理任务
        process_url = f"{BASE_URL}/restoration/process/{task_id}"
        response = requests.post(
            process_url,
            headers={"Authorization": f"Bearer {token}"}
        )
        
        if response.status_code != 200:
            print(f"  ❌ 提交失败: {response.text}")
            continue
        
        print(f"  ✓ 任务已提交")
        
        # 3. 等待处理完成
        status_url = f"{BASE_URL}/restoration/tasks/{task_id}"
        wait_start = time.time()
        
        while True:
            response = requests.get(
                status_url,
                headers={"Authorization": f"Bearer {token}"}
            )
            
            if response.status_code != 200:
                print(f"  ❌ 状态查询失败")
                break
            
            data = response.json()
            status = data.get("status")
            progress = data.get("progress", 0)
            
            if status == "completed":
                process_time = time.time() - wait_start
                total_time += process_time
                print(f"  ✓ 处理完成 (耗时: {process_time:.2f}s)")
                break
            elif status == "failed":
                print(f"  ❌ 处理失败: {data.get('message')}")
                break
            
            time.sleep(0.5)
    
    end_time = time.time()
    
    # 统计结果
    if len(test_files) > 0 and total_time > 0:
        avg_time = total_time / len(test_files)
        rate = 1 / avg_time
    else:
        avg_time = 0
        rate = 0
    
    print("\n" + "-"*60)
    print("  单文件串行处理统计(对照组)")
    print("-"*60)
    print(f"  测试文件数: {len(test_files)}")
    print(f"  总处理时间: {end_time - start_time:.2f}s")
    print(f"  纯处理时间: {total_time:.2f}s")
    print(f"  平均处理时间: {avg_time:.2f}s")
    print(f"  处理速率: {rate:.2f} 文件/秒")
    print("-"*60 + "\n")
    
    return {
        "total_time": end_time - start_time,
        "process_time": total_time,
        "avg_time": avg_time,
        "files_count": len(test_files),
        "rate": 1/avg_time
    }


def test_batch_processing(audio_files):
    """测试批量处理(优化组)"""
    print("\n" + "="*60)
    print("  步骤4: 批量处理测试(优化组)")
    print("="*60 + "\n")
    
    test_files = audio_files
    print(f"测试文件数量: {len(test_files)}")
    print(f"算法: {ALGORITHM}")
    print("\n开始批量上传处理...\n")
    
    start_time = time.time()
    
    # 1. 批量上传并触发批量处理
    upload_start = time.time()
    batch_upload_url = f"{BASE_URL}/restoration/batch/upload"
    
    files_list = []
    for file_path in test_files:
        filename = os.path.basename(file_path)
        files_list.append(("files", (filename, open(file_path, "rb"))))
    
    print(f"正在上传 {len(test_files)} 个文件...")
    
    response = requests.post(
        batch_upload_url,
        headers={"Authorization": f"Bearer {token}"},
        files=files_list,
        data={"algorithm": ALGORITHM}
    )
    
    upload_time = time.time() - upload_start
    
    # 关闭所有文件句柄
    for _, (_, file_obj) in files_list:
        file_obj.close()
    
    if response.status_code != 200:
        print(f"❌ 批量上传失败: {response.text}")
        return None
    
    data = response.json()
    batch_id = data["batch_id"]
    print(f"✓ 批量上传成功 (耗时: {upload_time:.2f}s)")
    print(f"  Batch ID: {batch_id}")
    print(f"  文件数量: {data['count']}")
    
    # 2. 监控批量处理进度
    status_url = f"{BASE_URL}/restoration/batch/tasks/{batch_id}"
    process_start = time.time()
    
    print("\n监控批量处理进度...")
    
    while True:
        response = requests.get(
            status_url,
            headers={"Authorization": f"Bearer {token}"}
        )
        
        if response.status_code != 200:
            print(f"❌ 状态查询失败")
            break
        
        data = response.json()
        status = data.get("status")
        progress = data.get("progress", 0)
        completed = data.get("completed_count", 0)
        total = data.get("total_count", 0)
        
        print(
            f"  [{datetime.now().strftime('%H:%M:%S')}] "
            f"进度: {progress:.1f}% | "
            f"完成: {completed}/{total} | "
            f"状态: {status}"
        )
        
        if status == "completed":
            process_time = time.time() - process_start
            end_time = time.time()
            
            print(f"\n✓ 批量处理完成!")
            print(f"  处理耗时: {process_time:.2f}s")
            print(f"  总耗时: {end_time - start_time:.2f}s")
            print(f"  平均耗时: {process_time / total:.2f}s")
            print(f"  处理速率: {total / process_time:.2f} 文件/秒")
            
            # 显示结果详情
            print("\n结果详情:")
            results = data.get("results", [])
            
            success_count = len([r for r in results if r.get("status") == "completed"])
            failed_count = len(results) - success_count
            
            print(f"  成功: {success_count}/{total}")
            print(f"  失败: {failed_count}")
            
            # 显示处理时间统计
            if success_count > 0:
                process_times = [r.get("processing_time", 0) for r in results if r.get("status") == "completed"]
                avg_process_time = sum(process_times) / len(process_times)
                print(f"  平均单文件处理时间: {avg_process_time:.2f}s")
            
            break
        elif status == "failed":
            print(f"❌ 批量处理失败: {data.get('message')}")
            break
        
        time.sleep(1)
    
    # 3. 检查GPU状态
    print("\n" + "-"*60)
    print("  GPU显存状态")
    print("-"*60)
    
    gpu_status_url = f"{BASE_URL}/restoration/gpu-status"
    response = requests.get(
        gpu_status_url,
        headers={"Authorization": f"Bearer {token}"}
    )
    
    if response.status_code == 200:
        gpu_data = response.json()
        if gpu_data.get("gpu_monitor"):
            print(f"  GPU监控: ✅ 已启动")
            print(f"  设备: {gpu_data.get('device_name')}")
            print(f"  显存占用: {gpu_data.get('allocated_mb'):.1f}MB")
            print(f"  显存预留: {gpu_data.get('reserved_mb'):.1f}MB")
            print(f"  利用率: {gpu_data.get('utilization_pct'):.1f}%")
            print(f"  状态: {gpu_data.get('status')}")
        else:
            print(f"  GPU监控: ❌ 未启动")
    
    print("-"*60 + "\n")
    
    return {
        "total_time": end_time - start_time,
        "upload_time": upload_time,
        "process_time": process_time,
        "avg_time": process_time / total,
        "files_count": total,
        "rate": total / process_time
    }


def compare_performance(single_result, batch_result):
    """对比性能数据"""
    print("\n" + "="*60)
    print("  性能对比分析")
    print("="*60 + "\n")
    
    if single_result and batch_result:
        print("对比表格:")
        print("-"*60)
        print(f"{'指标':<20} {'单文件串行':<15} {'批量处理':<15} {'提升':<10}")
        print("-"*60)
        
        # 计算提升率
        total_improvement = (
            (single_result["total_time"] - batch_result["total_time"]) 
            / single_result["total_time"] * 100
        )
        
        avg_improvement = (
            (single_result["avg_time"] - batch_result["avg_time"]) 
            / single_result["avg_time"] * 100
        )
        
        rate_improvement = (
            (batch_result["rate"] - single_result["rate"]) 
            / single_result["rate"] * 100
        )
        
        print(f"{'测试文件数':<20} {single_result['files_count']:<15} {batch_result['files_count']:<15} {'-':<10}")
        print(f"{'总处理时间(s)':<20} {single_result['total_time']:<15.2f} {batch_result['total_time']:<15.2f} {total_improvement:<10.1f}%")
        print(f"{'平均处理时间(s)':<20} {single_result['avg_time']:<15.2f} {batch_result['avg_time']:<15.2f} {avg_improvement:<10.1f}%")
        print(f"{'处理速率(文件/秒)':<20} {single_result['rate']:<15.2f} {batch_result['rate']:<15.2f} {rate_improvement:<10.1f}%")
        
        print("-"*60)
        
        print("\n关键发现:")
        print(f"  ✓ 批量处理节省总时间: {single_result['total_time'] - batch_result['total_time']:.2f}s")
        print(f"  ✓ 平均处理速度提升: {avg_improvement:.1f}%")
        print(f"  ✓ 处理吞吐量提升: {rate_improvement:.1f}%")
        
        # 预估全量40文件对比
        if batch_result["files_count"] == 40:
            print("\n40文件全量测试结果:")
            print(f"  批量处理总耗时: {batch_result['total_time']:.2f}s")
            estimated_single_time = single_result['avg_time'] * 40
            print(f"  串行处理预估耗时: {estimated_single_time:.2f}s")
            print(f"  节省时间: {estimated_single_time - batch_result['total_time']:.2f}s")
        
    else:
        print("⚠️  性能对比数据不完整")
    
    print("="*60 + "\n")


def main():
    """主测试流程"""
    print("\n" + "="*60)
    print("  AudioMOS 批量音频修复性能测试")
    print("="*60)
    print(f"  测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  音频目录: {AUDIO_DIR}")
    print(f"  算法: {ALGORITHM}")
    print("="*60 + "\n")
    
    try:
        # 步骤1: 登录
        login()  # 自动设置全局token变量
        
        # 步骤2: 准备测试文件
        audio_files = get_audio_files(TEST_FILES_COUNT)
        
        if len(audio_files) < TEST_FILES_COUNT:
            print(f"⚠️  音频文件数量不足({len(audio_files)}/{TEST_FILES_COUNT})")
            return
        
        # 步骤3: 单文件串行处理测试(对照组)
        single_result = test_single_file_processing(audio_files)
        
        # 步骤4: 批量处理测试(优化组)
        batch_result = test_batch_processing(audio_files)
        
        # 步骤5: 性能对比
        compare_performance(single_result, batch_result)
        
        # 步骤6: 保存测试报告
        print("\n" + "="*60)
        print("  测试完成")
        print("="*60)
        print(f"  完成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*60 + "\n")
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()