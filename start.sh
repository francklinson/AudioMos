#!/bin/bash

# AudioMOS 服务管理脚本
# 2026-05-12

# 获取脚本所在目录的绝对路径
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
cd "$SCRIPT_DIR"

# 读取配置文件中的端口
CONFIG_FILE="$SCRIPT_DIR/config/config.yaml"

# 从 YAML 文件读取端口的函数
read_yaml_value() {
    local file="$1"
    local key="$2"
    local default="$3"
    
    if [ -f "$file" ]; then
        # 使用 grep 和 sed 提取 YAML 值
        local value=$(grep -E "^\s*$key:" "$file" | head -1 | sed 's/.*:\s*//' | sed 's/"//g' | tr -d ' ')
        if [ -n "$value" ]; then
            echo "$value"
            return
        fi
    fi
    echo "$default"
}

# 读取嵌套 YAML 值 (如 server.backend.port)
read_nested_yaml() {
    local file="$1"
    local section="$2"
    local subsection="$3"
    local key="$4"
    local default="$5"
    
    if [ -f "$file" ]; then
        # 找到 section 下的 subsection 下的 key
        local in_section=false
        local in_subsection=false
        
        while IFS= read -r line; do
            # 检查是否进入目标 section
            if echo "$line" | grep -qE "^$section:"; then
                in_section=true
                continue
            fi
            
            # 如果在 section 中，检查是否进入 subsection
            if $in_section && echo "$line" | grep -qE "^\s+$subsection:"; then
                in_subsection=true
                continue
            fi
            
            # 如果在 subsection 中，查找 key
            if $in_subsection && echo "$line" | grep -qE "^\s+$key:"; then
                local value=$(echo "$line" | sed 's/.*:\s*//' | sed 's/"//g' | tr -d ' ')
                if [ -n "$value" ]; then
                    echo "$value"
                    return
                fi
            fi
            
            # 如果缩进减少，退出 subsection
            if $in_subsection && echo "$line" | grep -qE "^\S"; then
                in_subsection=false
            fi
            
            # 如果缩进减少，退出 section
            if $in_section && echo "$line" | grep -qE "^\S" && ! echo "$line" | grep -qE "^$section:"; then
                in_section=false
            fi
        done < "$file"
    fi
    
    echo "$default"
}

# 检测 host 是否支持的函数
check_host_support() {
    local host="$1"
    local port="${2:-8000}"
    
    # 使用 Python 检测，并捕获退出码
    python3 << EOF
import socket
import sys
try:
    socket.getaddrinfo('$host', $port, socket.AF_INET, socket.SOCK_STREAM)
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind(('$host', 0))
    s.close()
    sys.exit(0)
except Exception as e:
    sys.exit(1)
EOF
    local exit_code=$?
    return $exit_code
}

# 获取最佳 host
get_best_host() {
    # 优先尝试 127.0.0.1，如果不支持则尝试 0.0.0.0
    if check_host_support "127.0.0.1"; then
        echo "127.0.0.1"
    elif check_host_support "0.0.0.0"; then
        echo "0.0.0.0"
    else
        echo "127.0.0.1"  # 默认返回最安全的
    fi
}

# 获取服务器IP的函数
get_server_ip() {
    # 方法1: 通过UDP连接外部地址获取
    local ip=$(python3 -c "
import socket
import sys
try:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.settimeout(2)
    s.connect(('8.8.8.8', 80))
    ip = s.getsockname()[0]
    s.close()
    print(ip)
    sys.exit(0)
except:
    sys.exit(1)
" 2>/dev/null)
    
    if [ -n "$ip" ]; then
        echo "$ip"
        return
    fi
    
    # 方法2: 通过hostname获取
    local hostname=$(hostname)
    ip=$(python3 -c "
import socket
import sys
try:
    ip = socket.gethostbyname('$hostname')
    if ip and ip != '127.0.0.1':
        print(ip)
        sys.exit(0)
except:
    pass
sys.exit(1)
" 2>/dev/null)
    
    echo "$ip"
}

# 获取适合服务器的最佳host
get_best_host_for_server() {
    # 优先尝试 0.0.0.0
    if check_host_support "0.0.0.0"; then
        echo "0.0.0.0"
        return
    fi
    
    # 尝试获取服务器实际IP
    local server_ip=$(get_server_ip)
    if [ -n "$server_ip" ] && check_host_support "$server_ip"; then
        echo "$server_ip"
        return
    fi
    
    # 最后使用 127.0.0.1
    echo "127.0.0.1"
}

# 验证并修复 host 配置
validate_host() {
    local host="$1"
    local service_name="$2"
    
    # 如果配置的是 "auto"，自动选择最佳host
    if [ "$host" = "auto" ] || [ "$host" = "AUTO" ] || [ "$host" = "Auto" ]; then
        local new_host=$(get_best_host_for_server)
        if [ "$new_host" = "0.0.0.0" ]; then
            echo "ℹ️  $service_name 使用 auto 模式，自动选择 host: $new_host (支持外部访问)" >&2
        elif [ "$new_host" != "127.0.0.1" ] && [ "$new_host" != "localhost" ]; then
            echo "ℹ️  $service_name 使用 auto 模式，自动选择 host: $new_host (服务器实际IP)" >&2
        else
            echo "⚠️  $service_name 使用 auto 模式，但无法获取公网IP，使用: $new_host (仅本地访问)" >&2
        fi
        echo "$new_host"
        return
    fi
    
    # 如果配置的是 0.0.0.0，检查是否支持
    if [ "$host" = "0.0.0.0" ]; then
        if ! check_host_support "0.0.0.0"; then
            # 尝试获取服务器实际IP
            local server_ip=$(get_server_ip)
            if [ -n "$server_ip" ] && check_host_support "$server_ip"; then
                echo "⚠️  当前系统不支持 0.0.0.0，已将 $service_name 的 host 自动调整为服务器实际IP: $server_ip" >&2
                echo "   如需使用 0.0.0.0，请检查 /etc/hosts 或系统网络配置" >&2
                echo "$server_ip"
                return
            else
                local new_host="127.0.0.1"
                echo "⚠️  当前系统不支持 0.0.0.0，且无法获取服务器IP，已将 $service_name 的 host 自动调整为 $new_host" >&2
                echo "   注意: 此配置仅允许本机访问！" >&2
                echo "$new_host"
                return
            fi
        else
            # 支持 0.0.0.0，直接使用
            echo "$host"
            return
        fi
    fi
    
    # 检查配置的 host 是否支持
    if ! check_host_support "$host"; then
        local new_host=$(get_best_host_for_server)
        echo "⚠️  配置的 host '$host' 在当前系统不可用，已将 $service_name 的 host 自动调整为 $new_host" >&2
        echo "$new_host"
        return
    fi
    
    # 配置正常，直接使用
    echo "$host"
}

# 读取配置（优先使用环境变量，其次配置文件，最后默认值）
BACKEND_HOST="${AUDIOMOS_HOST:-$(read_yaml_value "$CONFIG_FILE" "host" "0.0.0.0")}"
BACKEND_PORT="${AUDIOMOS_PORT:-$(read_yaml_value "$CONFIG_FILE" "port" "8002")}"

echo ""
echo "================================"
echo "  服务配置"
echo "================================"
echo ""
echo "  服务地址: $BACKEND_HOST:$BACKEND_PORT"
echo ""

# 如果配置的是 auto，进行自动检测
if [ "$BACKEND_HOST" = "auto" ]; then
    BACKEND_HOST=$(validate_host "$BACKEND_HOST" "AudioMOS")
fi

echo ""

# PID 文件路径
BACKEND_PID_FILE="$SCRIPT_DIR/.backend.pid"

# 显示帮助信息
show_help() {
    echo "================================"
    echo "  AudioMOS 服务管理脚本"
    echo "================================"
    echo ""
    echo "用法: ./start.sh [命令] [选项]"
    echo ""
    echo "命令:"
    echo "  start           启动服务"
    echo "  stop            停止所有服务"
    echo "  restart         重启服务"
    echo "  status          查看服务状态"
    echo "  models          检查模型文件状态"
    echo "  help            显示帮助信息"
    echo ""
    echo "选项:"
    echo "  --port <port>   指定端口"
    echo "  --host <host>   指定地址"
    echo ""
    echo ""
    echo "当前配置:"
    echo "  后端: $BACKEND_HOST:$BACKEND_PORT"
    echo ""
    echo "环境变量(部署时设置):"
    echo "  AUDIOMOS_HOST             服务主机地址"
    echo "  AUDIOMOS_PORT             服务端口"
    echo "  AUDIOMOS_SECRET_KEY       JWT密钥 (部署时必改)"
    echo "  AUDIOMOS_ADMIN_PASSWORD   管理员密码"
    echo "  AUDIOMOS_CORS_ORIGINS     CORS允许来源,逗号分隔"
    echo "  AUDIOMOS_REF_DIR          参考音频目录"
    echo "  AUDIOMOS_UPLOAD_DIR       上传目录"
    echo "  AUDIOMOS_RESULT_DIR       结果目录"
    echo "  AUDIOMOS_LOG_LEVEL        日志级别"
    echo "  AUDIOMOS_CUDA_ENABLED     CUDA启用(true/false)"
    echo "  CUDA_VISIBLE_DEVICES      GPU设备ID"
    echo ""
    echo "快速安全启动示例:"
    echo '  export AUDIOMOS_SECRET_KEY="$(openssl rand -hex 32)"'
    echo '  export AUDIOMOS_ADMIN_PASSWORD="YourP@ss123"'
    echo '  export AUDIOMOS_CORS_ORIGINS="https://your-domain.com"'
    echo "  bash start.sh start"
    echo ""
}

# 检查服务是否正在运行
check_status() {
    local backend_running=false
    local unified_pid=""

    # 检查一体服务 PID（优先）
    if [ -f "$SCRIPT_DIR/.unified.pid" ]; then
        unified_pid=$(cat "$SCRIPT_DIR/.unified.pid")
        if ps -p "$unified_pid" > /dev/null 2>&1; then
            backend_running=true
        fi
    fi

    # 回退检查传统后端 PID
    if ! $backend_running && [ -f "$BACKEND_PID_FILE" ]; then
        local backend_pid=$(cat "$BACKEND_PID_FILE")
        if ps -p "$backend_pid" > /dev/null 2>&1; then
            backend_running=true
        fi
    fi

    if $backend_running; then
        echo "running"
    else
        echo "stopped"
    fi
}

# 显示服务状态
show_status() {
    echo "================================"
    echo "  AudioMOS 服务状态"
    echo "================================"
    echo ""

    local status=$(check_status)

    if [ "$status" = "running" ]; then
        local _pid="$(cat "$SCRIPT_DIR/.unified.pid" 2>/dev/null || cat "$BACKEND_PID_FILE" 2>/dev/null || echo "?")"
        echo "✅ AudioMOS 服务: 运行中 (PID: $_pid)"
        echo "   地址: http://$BACKEND_HOST:$BACKEND_PORT"
        echo ""
        echo "📡 前端页面: http://$BACKEND_HOST:$BACKEND_PORT"
        echo "📚 API文档:  http://$BACKEND_HOST:$BACKEND_PORT/docs"
    elif [ "$status" = "backend_only" ]; then
        local _pid="$(cat "$BACKEND_PID_FILE" 2>/dev/null || echo "?")"
        echo "✅ 后端服务: 运行中 (PID: $_pid)"
        echo "   地址: http://$BACKEND_HOST:$BACKEND_PORT"
        echo ""
        echo "❌ 后端服务: 未运行"
        echo ""
    else
        echo "❌ 后端服务: 未运行"
    fi
    echo ""
}

# 检查模型文件
check_models() {
    echo "================================"
    echo "  检查模型文件"
    echo "================================"
    echo ""

    local all_ready=true

    # 1. 检查TCF模型 (音色还原度) - 多模型检查
    echo "🔍 检查 TCF (音色还原度) 模型..."
    echo "   说明: 使用多模型加权评估音色还原度"
    echo "   模型列表: eres2net/campplus/ecapa-tdnn/res2net/resnet34"
    echo ""
    echo "   检查路径:"
    echo "      项目路径: $SCRIPT_DIR/models/tcf/"
    echo "      缓存路径: $HOME/.cache/modelscope/hub/"
    echo ""

    local tcf_models=("eres2net" "campplus" "ecapa-tdnn" "res2net" "resnet34")
    local tcf_model_ids=("damo/speech_eres2net_sv_zh-cn_16k-common" "damo/speech_campplus_sv_zh-cn_16k-common" "damo/speech_ecapa-tdnn_sv_zh-cn_cnceleb_16k" "damo/speech_res2net_sv_zh-cn_3dspeaker_16k" "damo/speech_resnet34_sv_zh-cn_3dspeaker_16k")
    local tcf_available=0
    local tcf_total=${#tcf_models[@]}

    for i in "${!tcf_models[@]}"; do
        local model_name="${tcf_models[$i]}"
        local model_id="${tcf_model_ids[$i]}"
        local project_path="$SCRIPT_DIR/models/tcf/$model_name/configuration.json"
        local cache_path="$HOME/.cache/modelscope/hub/$model_id/configuration.json"

        if [ -f "$project_path" ]; then
            echo "   ✅ $model_name"
            echo "      来源: 项目路径"
            echo "      位置: $SCRIPT_DIR/models/tcf/$model_name/"
            ((tcf_available++))
        elif [ -f "$cache_path" ]; then
            echo "   ✅ $model_name"
            echo "      来源: 本地缓存"
            echo "      位置: $HOME/.cache/modelscope/hub/$model_id/"
            ((tcf_available++))
        else
            echo "   ❌ $model_name"
            echo "      状态: 缺失"
            echo "      期望路径: $SCRIPT_DIR/models/tcf/$model_name/"
            echo "      模型ID: $model_id"
        fi
    done

    echo ""
    echo "   汇总:"
    if [ $tcf_available -eq 0 ]; then
        echo "      ⚠️  所有TCF模型都缺失"
        all_ready=false
    elif [ $tcf_available -lt $tcf_total ]; then
        echo "      ⚠️  TCF模型部分可用 ($tcf_available/$tcf_total)"
        echo "      说明: 将使用可用模型进行加权评估"
    else
        echo "      ✅ 所有TCF模型已就绪 ($tcf_available/$tcf_total)"
    fi
    echo ""

    # 2. 检查WeNet模型 (WER语音识别) - 优先检查项目路径
    echo "🔍 检查 WeNet (语音识别) 模型..."
    echo "   说明: 用于计算WER(词错误率)"
    echo ""
    echo "   检查路径:"
    echo "      项目路径: $SCRIPT_DIR/models/wenet/"
    echo "      缓存路径: $HOME/.wenet/wenetspeech/"
    echo ""

    local wenet_project_path="$SCRIPT_DIR/models/wenet/final.pt"
    local wenet_cache_path="$HOME/.wenet/wenetspeech/final.pt"

    if [ -f "$wenet_project_path" ]; then
        echo "   ✅ WeNet模型已就绪"
        echo "      来源: 项目路径"
        echo "      位置: $SCRIPT_DIR/models/wenet/"
        echo "      主文件: final.pt"
    elif [ -f "$wenet_cache_path" ]; then
        echo "   ✅ WeNet模型已就绪"
        echo "      来源: 本地缓存"
        echo "      位置: $HOME/.wenet/wenetspeech/"
        echo "      主文件: final.pt"
    else
        echo "   ❌ WeNet模型缺失"
        echo "      状态: 未找到"
        echo "      期望路径(项目): $SCRIPT_DIR/models/wenet/final.pt"
        echo "      期望路径(缓存): $HOME/.wenet/wenetspeech/final.pt"
        echo "      下载命令: wenet.load_model('wenetspeech') 会自动下载"
        all_ready=false
    fi
    echo ""

    # 3. 检查NISQA模型
    echo "🔍 检查 NISQA 模型..."
    echo "   说明: 语音质量评估模型"
    echo ""
    echo "   检查路径:"
    echo "      项目路径: $SCRIPT_DIR/models/nisqa/weights/"
    echo "      算法路径: $SCRIPT_DIR/app/algorithms/nisqa/weights/"
    echo ""

    local nisqa_model_path="$SCRIPT_DIR/models/nisqa/weights/nisqa.tar"
    local nisqa_model_path_old="$SCRIPT_DIR/app/algorithms/nisqa/weights/nisqa.tar"

    if [ -f "$nisqa_model_path" ]; then
        echo "   ✅ NISQA模型已就绪"
        echo "      来源: 项目路径"
        echo "      位置: $nisqa_model_path"
        local nisqa_size=$(du -sh "$nisqa_model_path" 2>/dev/null | cut -f1)
        echo "      大小: $nisqa_size"
    elif [ -f "$nisqa_model_path_old" ]; then
        echo "   ✅ NISQA模型已就绪"
        echo "      来源: 算法路径"
        echo "      位置: $nisqa_model_path_old"
        local nisqa_size=$(du -sh "$nisqa_model_path_old" 2>/dev/null | cut -f1)
        echo "      大小: $nisqa_size"
    else
        echo "   ⚠️  NISQA模型缺失 (可选依赖)"
        echo "      状态: 未找到"
        echo "      期望路径(项目): $nisqa_model_path"
        echo "      期望路径(算法): $nisqa_model_path_old"
        echo "      说明: NISQA是可选依赖，缺失时不影响核心功能"
    fi
    echo ""

    # 4. 检查DNSMOS模型
    echo "🔍 检查 DNSMOS 模型..."
    echo "   说明: DNSMOS语音质量评估 (P808 + Primary)"
    echo ""
    echo "   检查路径:"
    echo "      项目路径: $SCRIPT_DIR/models/dnsmos/"
    echo "      算法路径: $SCRIPT_DIR/app/algorithms/dnsmos/"
    echo ""

    local dnsmos_p808_path="$SCRIPT_DIR/models/dnsmos/DNSMOS/model_v8.onnx"
    local dnsmos_primary_path="$SCRIPT_DIR/models/dnsmos/pDNSMOS/sig_bak_ovr.onnx"
    local dnsmos_p808_path_old="$SCRIPT_DIR/app/algorithms/dnsmos/DNSMOS/model_v8.onnx"
    local dnsmos_primary_path_old="$SCRIPT_DIR/app/algorithms/dnsmos/pDNSMOS/sig_bak_ovr.onnx"
    local dnsmos_ok=true

    echo "   检查子模型:"

    # 检查P808模型
    if [ -f "$dnsmos_p808_path" ]; then
        echo "      ✅ P808模型"
        echo "         来源: 项目路径"
        echo "         位置: $dnsmos_p808_path"
        local p808_size=$(du -sh "$dnsmos_p808_path" 2>/dev/null | cut -f1)
        echo "         大小: $p808_size"
    elif [ -f "$dnsmos_p808_path_old" ]; then
        echo "      ✅ P808模型"
        echo "         来源: 算法路径"
        echo "         位置: $dnsmos_p808_path_old"
        local p808_size=$(du -sh "$dnsmos_p808_path_old" 2>/dev/null | cut -f1)
        echo "         大小: $p808_size"
    else
        echo "      ⚠️  P808模型缺失 (可选)"
        echo "         期望路径(项目): $dnsmos_p808_path"
        echo "         期望路径(算法): $dnsmos_p808_path_old"
        dnsmos_ok=false
    fi

    # 检查Primary模型
    if [ -f "$dnsmos_primary_path" ]; then
        echo "      ✅ Primary模型"
        echo "         来源: 项目路径"
        echo "         位置: $dnsmos_primary_path"
        local primary_size=$(du -sh "$dnsmos_primary_path" 2>/dev/null | cut -f1)
        echo "         大小: $primary_size"
    elif [ -f "$dnsmos_primary_path_old" ]; then
        echo "      ✅ Primary模型"
        echo "         来源: 算法路径"
        echo "         位置: $dnsmos_primary_path_old"
        local primary_size=$(du -sh "$dnsmos_primary_path_old" 2>/dev/null | cut -f1)
        echo "         大小: $primary_size"
    else
        echo "      ⚠️  Primary模型缺失 (可选)"
        echo "         期望路径(项目): $dnsmos_primary_path"
        echo "         期望路径(算法): $dnsmos_primary_path_old"
        dnsmos_ok=false
    fi

    echo ""
    echo "   汇总:"
    if $dnsmos_ok; then
        echo "      ✅ DNSMOS模型已就绪"
    else
        echo "      ⚠️  DNSMOS部分模型缺失 (可选依赖)"
        echo "      说明: DNSMOS是可选依赖，缺失时不影响核心功能"
    fi
    echo ""

    # 5. 检查Scoreq模型 (通过Python检查)
    echo "🔍 检查 Scoreq 模块..."
    echo "   说明: 语音质量评估模块"
    echo ""
    if source .venv/bin/activate && python -c "import scoreq; print('OK')" 2>/dev/null | grep -q "OK"; then
        echo "   ✅ Scoreq模块已安装"
        local scoreq_path=$(source .venv/bin/activate && python -c "import scoreq; print(scoreq.__file__)" 2>/dev/null)
        echo "      来源: Python包"
        echo "      位置: $scoreq_path"
    else
        echo "   ⚠️  Scoreq模块未安装 (可选依赖)"
        echo "      状态: 未安装"
        echo "      说明: Scoreq是可选依赖，缺失时不影响核心功能"
    fi
    echo ""

    # 6. 检查speechmetrics (通过Python检查)
    echo "🔍 检查 SpeechMetrics 模块..."
    echo "   说明: 语音质量评估指标库 (STOI/SISDR等)"
    echo ""
    if source .venv/bin/activate && PYTHONPATH="$SCRIPT_DIR/app/algorithms:$PYTHONPATH" python -c "import speechmetrics; print('OK')" 2>/dev/null | grep -q "OK"; then
        echo "   ✅ SpeechMetrics模块已就绪"
        echo "      来源: 项目算法目录"
        echo "      位置: $SCRIPT_DIR/app/algorithms/speechmetrics/"
    else
        echo "   ⚠️  SpeechMetrics模块未就绪 (可选依赖)"
        echo "      状态: 未就绪"
        echo "      说明: SpeechMetrics是可选依赖，缺失时不影响核心功能"
    fi
    echo ""

    # 7. 检查UTMOS模型
    echo "🔍 检查 UTMOS 模型..."
    echo "   说明: UTokyo-SaruLab MOS预测系统 (VoiceMOS 2024第一名)"
    echo ""
    echo "   检查路径:"
    echo "      项目路径: $SCRIPT_DIR/models/utmos/models/fusion_stage3/"
    echo ""

    local utmos_model_dir="$SCRIPT_DIR/models/utmos/models/fusion_stage3"
    local utmos_ok=true

    if [ -d "$utmos_model_dir" ]; then
        # 只统计大于 0 的有效模型文件
        local utmos_count=$(find "$utmos_model_dir" -name "*.pth" -size +0 | wc -l)
        echo "   模型文件:"
        for f in "$utmos_model_dir"/fold*.pth; do
            if [ -f "$f" ]; then
                local fname=$(basename "$f")
                local fsize=$(du -sh "$f" 2>/dev/null | cut -f1)
                # 检查文件大小是否为 0
                if [ "$(stat -c%s "$f" 2>/dev/null)" -gt 0 ]; then
                    echo "      ✅ $fname ($fsize)"
                else
                    echo "      ❌ $fname (0 字节 - 无效文件)"
                fi
            fi
        done
        echo ""
        if [ $utmos_count -ge 5 ]; then
            echo "   ✅ UTMOSv2模型已就绪 ($utmos_count个fold)"
            echo "      来源: 项目路径"
            echo "      位置: $utmos_model_dir"
        else
            echo "   ⚠️  UTMOSv2模型不完整 ($utmos_count/5个fold)"
            echo "      位置: $utmos_model_dir"
            utmos_ok=false
        fi
    else
        echo "   ❌ UTMOSv2模型缺失"
        echo "      状态: 未找到"
        echo "      期望路径: $utmos_model_dir"
        echo "      下载命令: python download_utmos_models.py"
        utmos_ok=false
    fi

    echo ""
    # 检查UTMOS模块
    if source .venv/bin/activate && PYTHONPATH="$SCRIPT_DIR/app/algorithms:$PYTHONPATH" python -c "import utmosv2; print('OK')" 2>/dev/null | grep -q "OK"; then
        echo "   ✅ UTMOS模块已安装"
        echo "      来源: 项目算法目录"
        echo "      位置: $SCRIPT_DIR/app/algorithms/utmos/"
    else
        echo "   ⚠️  UTMOS模块未安装 (可选依赖)"
        echo "      安装命令: pip install -e ./app/algorithms/utmos"
    fi
    echo ""

    # 8. 检查wav2vec2模型 (UTMOS依赖)
    echo "🔍 检查 wav2vec2 模型 (UTMOS依赖)..."
    echo "   说明: facebook/wav2vec2-base (UTMOS的SSL编码器依赖)"
    echo ""
    echo "   检查路径:"
    echo "      项目路径: $SCRIPT_DIR/models/wav2vec2/facebook--wav2vec2-base/"
    echo ""

    local wav2vec2_path="$SCRIPT_DIR/models/wav2vec2/facebook--wav2vec2-base"

    if [ -d "$wav2vec2_path" ] && [ -f "$wav2vec2_path/model.safetensors" ]; then
        echo "   ✅ wav2vec2-base模型已就绪"
        echo "      来源: 项目路径"
        echo "      位置: $wav2vec2_path"
        local wav2vec2_size=$(du -sh "$wav2vec2_path" 2>/dev/null | cut -f1)
        echo "      总大小: $wav2vec2_size"
        if [ -f "$wav2vec2_path/model.safetensors" ]; then
            local model_size=$(du -sh "$wav2vec2_path/model.safetensors" 2>/dev/null | cut -f1)
            echo "      模型文件: model.safetensors ($model_size)"
        fi
    else
        echo "   ❌ wav2vec2-base模型缺失"
        echo "      状态: 未找到"
        echo "      期望路径: $wav2vec2_path"
        echo "      下载命令: python download_wav2vec2_models.py"
        all_ready=false
    fi
    echo ""

    # 9. 检查timm模型 (UTMOS依赖)
    echo "🔍 检查 timm 模型 (UTMOS依赖)..."
    echo "   说明: tf_efficientnetv2_s.in21k_ft_in1k (UTMOS的多谱图模型依赖)"
    echo ""
    echo "   检查路径:"
    echo "      项目路径: $SCRIPT_DIR/models/timm/"
    echo ""

    local timm_model_name="tf_efficientnetv2_s.in21k_ft_in1k"
    local timm_model_path="$SCRIPT_DIR/models/timm/${timm_model_name}.safetensors"

    if [ -f "$timm_model_path" ]; then
        echo "   ✅ timm模型已就绪"
        echo "      模型名称: $timm_model_name"
        echo "      来源: 项目路径"
        echo "      位置: $timm_model_path"
        local timm_size=$(du -sh "$timm_model_path" 2>/dev/null | cut -f1)
        echo "      大小: $timm_size"
    else
        echo "   ❌ timm模型缺失"
        echo "      状态: 未找到"
        echo "      模型名称: $timm_model_name"
        echo "      期望路径: $timm_model_path"
        echo "      说明: 该模型用于UTMOS的多谱图特征提取"
        all_ready=false
    fi
    echo ""

    # ========================
    # 10. 检查降噪算法环境与模型
    # ========================
    echo "🔍 检查降噪算法环境与模型..."
    echo "   说明：包含 SpeechBrain / ClearerVoice 等"
    echo ""

    local denoise_all_ok=true

    # 10a. 检查 Python 依赖库
    echo "   📦 依赖库检查:"
    local denoise_deps=(
        "speechbrain:SpeechBrain 深度学习语音工具包"
        "torch:PyTorch 深度学习框架"
        "librosa:音频信号处理库"
        "soundfile:音频文件读写"
        "onnxruntime:ONNX Runtime 推理引擎"
        "pesq:PESQ 语音质量评估"
        "pystoi:STOI 短时可懂度评估"
        "clearvoice:ClearVoice 降噪推理平台(阿里达摩院)"
    )
    for dep_entry in "${denoise_deps[@]}"; do
        local dep_module="${dep_entry%%:*}"
        local dep_desc="${dep_entry##*:}"
        if source .venv/bin/activate && python -c "import $dep_module" 2>/dev/null; then
            echo "      ✅ $dep_module ($dep_desc)"
        else
            echo "      ❌ $dep_module ($dep_desc)"
            denoise_all_ok=false
            all_ready=false
        fi
    done

    # modelscope 单独检查（已知有 datasets 兼容性问题）
    if source .venv/bin/activate && python -c "
import datasets
if not hasattr(datasets, 'LargeList'):
    class _LargeListStub(list): pass
    datasets.LargeList = _LargeListStub
import modelscope
print('OK')
" 2>/dev/null | grep -q "OK"; then
        echo "      ✅ modelscope (ModelScope 模型库)"
    else
        echo "      ⚠️  modelscope (ModelScope 模型库) — 已打兼容补丁，按需使用"
    fi

    echo ""

    # 10b. 检查 SpeechBrain 模型
    echo "   🧠 SpeechBrain 降噪模型:"
    local sb_model_dir="$SCRIPT_DIR/models/speechbrain"

    local sb_models=(
        "metricgan-plus-voicebank/enhance_model.ckpt:MetricGAN+ 语音增强"
        "sepformer-wham-enhancement/encoder.ckpt:SepFormer WHAM 语音分离"
        "sepformer-whamr-enhancement/encoder.ckpt:SepFormer WHAMR 语音分离"
    )
    for model_entry in "${sb_models[@]}"; do
        local model_file="${model_entry%%:*}"
        local model_desc="${model_entry##*:}"
        local full_path="$sb_model_dir/$model_file"
        if [ -f "$full_path" ]; then
            # 使用 -L 参数跟随符号链接，获取实际文件大小
            local model_dir="$(dirname "$full_path")"
            local model_size=$(du -shL "$model_dir" 2>/dev/null | cut -f1)
            echo "      ✅ $model_desc ($model_size)"
        else
            echo "      ⚠️  $model_desc — 首次使用自动下载"
        fi
    done
    echo ""

    # 10c. 检查 ClearVoice 模型 (FRCRN/MossFormer/MossFormer2)
    echo "   🎙️  ClearVoice 降噪模型 (阿里达摩院):"
    local cv_model_dir="$SCRIPT_DIR/models/clearvoice"

    # 检查全部 5 个 ClearVoice 模型
    local cv_models=(
        "FRCRN_SE_16K/last_best_checkpoint.pt:FRCRN 实时语音增强 (16kHz)"
        "MossFormerGAN_SE_16K/last_best_checkpoint.pt:MossFormerGAN 语音增强 (16kHz)"
        "MossFormer2_SS_16K/last_best_checkpoint.pt:MossFormer2 语音分离 (16kHz)"
        "MossFormer2_SE_48K/last_best_checkpoint.pt:MossFormer2 语音增强 (48kHz)"
        "MossFormer2_SR_48K/last_best_checkpoint_m.pt:MossFormer2 超分辨率 (48kHz)"
    )
    local cv_ok=true
    local cv_count=0
    local cv_total=5
    
    for model_entry in "${cv_models[@]}"; do
        local model_file="${model_entry%%:*}"
        local model_desc="${model_entry##*:}"
        local full_path="$cv_model_dir/$model_file"
        if [ -f "$full_path" ]; then
            local model_size=$(du -sh "$(dirname "$full_path")" 2>/dev/null | cut -f1)
            echo "      ✅ $model_desc — 已下载 ($model_size)"
            ((cv_count++))
        else
            echo "      ❌ $model_desc — 缺失"
            cv_ok=false
        fi
    done

    echo ""
    if [ $cv_count -eq $cv_total ]; then
        echo "      ✅ ClearVoice 模型已全部就绪 ($cv_count/$cv_total)"
        local total_size=$(du -sh "$cv_model_dir" 2>/dev/null | cut -f1)
        echo "      总大小：$total_size"
    elif [ $cv_count -gt 0 ]; then
        echo "      ⚠️  ClearVoice 模型部分可用 ($cv_count/$cv_total)"
        echo "      提示：运行 'python scripts/download_clearvoice_models.py' 下载缺失模型"
    else
        echo "      ❌ ClearVoice 模型全部缺失"
        echo "      提示：运行 'python scripts/download_clearvoice_models.py' 批量下载"
    fi
    echo ""

    # 10f. 检查降噪评估模块依赖
    echo "   📊 降噪评估模块依赖:"
    local eval_deps=(
        "pesq:PESQ 有参考语音质量评估"
        "pystoi:STOI 短时客观可懂度"
    )
    for dep_entry in "${eval_deps[@]}"; do
        local dep_module="${dep_entry%%:*}"
        local dep_desc="${dep_entry##*:}"
        if source .venv/bin/activate && python -c "import $dep_module" 2>/dev/null; then
            echo "      ✅ $dep_module ($dep_desc)"
        else
            echo "      ❌ $dep_module ($dep_desc)"
            denoise_all_ok=false
            all_ready=false
        fi
    done

    # UTMOS (已在上面检查过，这里确认模块路径)
    local utmos_denoise_ok=true
    if source .venv/bin/activate && python -c "
import sys
sys.path.insert(0, '$SCRIPT_DIR/app/algorithms')
from utmos.utmos_score import UTMOSCore
print('OK')
" 2>/dev/null | grep -q "OK"; then
        echo "      ✅ UTMOS 降噪评估模块"
    else
        echo "      ⚠️  UTMOS 评估模块不可用"
        utmos_denoise_ok=false
    fi

    # DNSMOS ONNX (降噪评估用)
    local dnsmos_onnx_dir="$SCRIPT_DIR/app/algorithms/dnsmos"
    if [ -f "$dnsmos_onnx_dir/pDNSMOS/sig_bak_ovr.onnx" ] && [ -f "$dnsmos_onnx_dir/DNSMOS/model_v8.onnx" ]; then
        echo "      ✅ DNSMOS ONNX 模型 (降噪评估)"
    else
        echo "      ⚠️  DNSMOS ONNX 模型不可用"
        utmos_denoise_ok=false
    fi

    # NISQA 降噪评估
    local nisqa_weight_path="$SCRIPT_DIR/app/algorithms/nisqa/weights/nisqa_3000.tar"
    if [ -f "$nisqa_weight_path" ]; then
        local nisqa_weight_size=$(du -sh "$nisqa_weight_path" 2>/dev/null | cut -f1)
        echo "      ✅ NISQA 权重 (降噪评估, $nisqa_weight_size)"
    else
        echo "      ⚠️  NISQA 权重不可用"
    fi
    echo ""

    # 10g. CUDA 检查
    echo "   🖥️  CUDA 设备检查:"
    if source .venv/bin/activate && python -c "
import torch
print(f'CUDA:{torch.cuda.is_available()}:{torch.cuda.device_count()}:{torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')
" 2>/dev/null | grep -q "CUDA:True"; then
        local cuda_info=$(source .venv/bin/activate && python -c "
import torch
print(f'{torch.cuda.get_device_name(0)} (CUDA {torch.version.cuda})')
" 2>/dev/null)
        echo "      ✅ CUDA 可用: $cuda_info"
    else
        echo "      ⚠️  CUDA 不可用，降噪算法将以 CPU 模式运行 (速度较慢)"
    fi
    echo ""

    # 降噪汇总
    echo "   降噪算法汇总:"
    if $denoise_all_ok; then
        echo "      ✅ 降噪算法核心依赖已就绪"
    else
        echo "      ⚠️  部分降噪算法依赖缺失 (详见上方检查)"
    fi
    echo ""

    # ========================
    # 汇总
    # ========================
    echo "================================"
    if $all_ready; then
        echo "  ✅ 所有核心模型已就绪"
    else
        echo "  ⚠️  部分模型缺失"
        echo ""
        echo "请运行以下命令下载缺失的模型:"
        echo "  python download_denoise_models.py"
    fi
    echo "================================"
    echo ""

    return $([ "$all_ready" = true ] && echo 0 || echo 1)
}

# 启动服务

# 停止服务
stop_services() {
    echo "================================"
    echo "  停止 AudioMOS 服务"
    echo "================================"
    echo ""

    local backend_stopped=false

    # 停止 PID 文件记录的后端进程
    if [ -f "$BACKEND_PID_FILE" ]; then
        local backend_pid=$(cat "$BACKEND_PID_FILE")
        if ps -p "$backend_pid" > /dev/null 2>&1; then
            echo "正在停止服务 (PID: $backend_pid)..."
            kill "$backend_pid" 2>/dev/null
            sleep 2
            if ps -p "$backend_pid" > /dev/null 2>&1; then
                kill -9 "$backend_pid" 2>/dev/null
            fi
            echo "✅ 服务已停止"
        else
            echo "服务未运行"
        fi
        rm -f "$BACKEND_PID_FILE"
        backend_stopped=true
    fi

    # 查找残留的 Python 后端进程
    local python_pids=$(pgrep -f "uvicorn.*app.main" 2>/dev/null)
    if [ -n "$python_pids" ]; then
        echo "发现残留进程，正在清理..."
        for pid in $python_pids; do
            echo "  停止进程 (PID: $pid)"
            kill -9 "$pid" 2>/dev/null || true
        done
        backend_stopped=true
    fi

    if ! $backend_stopped; then
        echo "没有运行中的服务"
    fi

    # 清理临时启动文件
    if [ -f "$SCRIPT_DIR/.start_server.py" ]; then
        rm -f "$SCRIPT_DIR/.start_server.py"
    fi

    # 按端口兜底清理（仅清理 LISTEN 状态的进程，避免误杀浏览器等连接方）
    local port="${AUDIOMOS_PORT:-$(read_yaml_value "$CONFIG_FILE" "port" "8002")}"
    local port_pids=$(lsof -ti :"$port" -sTCP:LISTEN 2>/dev/null)
    if [ -n "$port_pids" ]; then
        echo "清理端口 $port 残留进程..."
        for pid in $port_pids; do
            local pname=$(ps -p "$pid" -o comm= 2>/dev/null)
            echo "  清理 (PID: $pid, $pname)"
            kill -9 "$pid" 2>/dev/null || true
        done
        sleep 1
        if lsof -ti :"$port" -sTCP:LISTEN > /dev/null 2>&1; then
            echo "⚠️  警告: 端口 $port 仍有进程残留"
        else
            echo "✅ 端口 $port 已释放"
        fi
    fi

    echo "================================"
    echo ""
}

# 等待端口释放
wait_for_ports() {
    echo ""
    echo "等待端口释放..."
    for port in "${ports[@]}"; do
        local count=0
        while lsof -i :"$port" > /dev/null 2>&1 && [ $count -lt 10 ]; do
            sleep 1
            count=$((count + 1))
        done
        if [ $count -ge 10 ]; then
            echo "⚠️  端口 $port 可能仍被占用"
        else
            echo "✅ 端口 $port 已释放"
        fi
    done
}

# 启动前后端一体服务
start_unified() {
    # 解析参数
    local unified_port="$BACKEND_PORT"
    local unified_host="$BACKEND_HOST"
    
    # 解析 --port 和 --host 参数
    while [[ $# -gt 0 ]]; do
        case $1 in
            --port)
                unified_port="$2"
                shift 2
                ;;
            --host)
                unified_host="$2"
                shift 2
                ;;
            *)
                shift
                ;;
        esac
    done
    
    echo "================================"
    echo "  启动 AudioMOS 前后端一体服务"
    echo "================================"
    echo ""
    echo "项目路径: $SCRIPT_DIR"
    echo ""
    echo "配置:"
    echo "  监听地址: $unified_host:$unified_port"
    echo "  模式: 前后端一体（单服务）"
    echo ""
    
    # 检查虚拟环境
    if [ ! -d ".venv" ]; then
        echo "正在创建虚拟环境..."
        python3 -m venv .venv
    fi
    
    # 激活虚拟环境
    echo "激活虚拟环境..."
    source .venv/bin/activate
    
    # 检查依赖
    if ! pip show fastapi > /dev/null 2>&1; then
        echo "正在安装Python依赖..."
        pip install --upgrade pip
        pip install -r requirements.txt
    fi
    
    # 检查并自动构建前端
    local needs_build=false

    # 检测简化前端（HTML+JS+CSS，无需Node.js构建）
    if [ -f "$SCRIPT_DIR/backend/static/js/app.js" ] && [ -f "$SCRIPT_DIR/backend/static/css/app.css" ]; then
        echo "   ✅ 检测到简化前端（HTML+JS+CSS），跳过Node.js构建"
        needs_build=false
    elif [ ! -f "$SCRIPT_DIR/backend/static/index.html" ]; then
        echo ""
        echo "📦 未检测到前端构建文件，需要自动构建..."
        needs_build=true
    elif [ "$AUDIOMOS_REBUILD_FRONTEND" = "1" ] || [ "$AUDIOMOS_REBUILD_FRONTEND" = "true" ]; then
        echo ""
        echo "📦 已设置 AUDIOMOS_REBUILD_FRONTEND，强制重新构建前端..."
        needs_build=true
    elif [ "$SCRIPT_DIR/frontend/src" -nt "$SCRIPT_DIR/backend/static/index.html" ] && [ ! -f "$SCRIPT_DIR/backend/static/js/app.js" ]; then
        echo ""
        echo "📦 检测到前端源码有更新，自动重新构建..."
        needs_build=true
    fi

    if [ "$needs_build" = true ]; then
        # 检查 Node.js 和 npm
        if ! command -v node &> /dev/null; then
            echo "❌ 未检测到 Node.js，无法构建前端"
            echo "   请安装 Node.js (>=18): https://nodejs.org/"
            echo "   或手动构建后复制到 backend/static/"
            return 1
        fi
        if ! command -v npm &> /dev/null; then
            echo "❌ 未检测到 npm，无法构建前端"
            return 1
        fi

        echo "   Node.js: $(node --version)"
        echo "   npm:     $(npm --version)"

        cd "$SCRIPT_DIR/frontend"

        # 安装依赖
        if [ ! -d "node_modules" ] || [ "$AUDIOMOS_REBUILD_FRONTEND" = "1" ] || [ "$AUDIOMOS_REBUILD_FRONTEND" = "true" ]; then
            echo "   📥 安装前端依赖..."
            npm install --legacy-peer-deps || {
                echo "❌ npm install 失败"
                cd "$SCRIPT_DIR"
                return 1
            }
        else
            echo "   ✅ node_modules 已存在，跳过 npm install (设置 AUDIOMOS_REBUILD_FRONTEND=1 可强制重装)"
        fi

        # 构建
        echo "   🔨 构建前端..."
        npx vite build || {
            echo "❌ 前端构建失败"
            cd "$SCRIPT_DIR"
            return 1
        }

        # 复制到后端静态目录
        echo "   📋 复制构建产物到 backend/static/..."
        mkdir -p "$SCRIPT_DIR/backend/static"
        cp -r dist/* "$SCRIPT_DIR/backend/static/" || {
            echo "❌ 复制构建产物失败"
            cd "$SCRIPT_DIR"
            return 1
        }

        cd "$SCRIPT_DIR"
        echo "   ✅ 前端构建完成"
    else
        echo "   ✅ 前端已就绪 (设置 AUDIOMOS_REBUILD_FRONTEND=1 可强制重建)"
    fi
    
    # 检查模型文件
    echo ""
    check_models
    local models_status=$?
    
    if [ $models_status -ne 0 ]; then
        echo ""
        echo "⚠️  模型检查未通过,是否继续启动? (y/N)"
        read -r response
        if [[ ! "$response" =~ ^[Yy]$ ]]; then
            echo "启动已取消"
            return 1
        fi
        echo "继续启动(部分功能可能不可用)..."
        echo ""
    fi
    
    # 停止已有服务（两个都调用，覆盖所有启动方式）
    stop_services > /dev/null 2>&1
    stop_unified > /dev/null 2>&1

    # 启动一体服务
    echo ""
    echo "启动服务..."

    # ---- 部署环境变量配置 ----
    # 优先级: 环境变量(已设置) > 以下默认值 > 代码内兜底
    # 生产部署前设置:
    #   export AUDIOMOS_SECRET_KEY="$(openssl rand -hex 32)"
    #   export AUDIOMOS_ADMIN_PASSWORD="your-strong-password"
    #   export AUDIOMOS_CORS_ORIGINS="https://your-domain.com"
    export AUDIOMOS_HOST="${AUDIOMOS_HOST:-$unified_host}"
    export AUDIOMOS_PORT="${AUDIOMOS_PORT:-$unified_port}"
    # JWT密钥: 如未设置,从config.yaml读取; 再没有则使用安全随机生成(每次启动不同,踢下线已登录用户)
    if [ -z "$AUDIOMOS_SECRET_KEY" ]; then
        _cfg_key=$(read_yaml_value "$CONFIG_FILE" "secret_key" "")
        if [ -n "$_cfg_key" ] && [ "$_cfg_key" != "your-secret-key-change-this-in-production" ]; then
            export AUDIOMOS_SECRET_KEY="$_cfg_key"
        fi
    fi
    # 管理员密码: 如未设置,从config.yaml读取
    if [ -z "$AUDIOMOS_ADMIN_PASSWORD" ]; then
        _cfg_pwd=$(read_yaml_value "$CONFIG_FILE" "admin_password" "")
        if [ -n "$_cfg_pwd" ] && [ "$_cfg_pwd" != "tp123456" ]; then
            export AUDIOMOS_ADMIN_PASSWORD="$_cfg_pwd"
        fi
    fi
    # CORS来源: 如未设置且非生产默认,保持通配
    if [ -z "$AUDIOMOS_CORS_ORIGINS" ] && [ "$AUDIOMOS_CORS_ORIGINS_AUTO" != "1" ]; then
        :  # 保持未设置,代码默认为"*"
    fi

    cd "$SCRIPT_DIR/backend"
    
    # 使用临时Python文件启动，避免shell引号转义问题
    cat > "$SCRIPT_DIR/.start_server.py" << 'PYEOF'
import sys
import os

_script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(os.path.join(_script_dir, 'backend'))
sys.path.insert(0, '.')

import uvicorn
from app.core.logging_config import logger

host = os.environ.get('AUDIOMOS_HOST', '0.0.0.0')
port = int(os.environ.get('AUDIOMOS_PORT', '8002'))

logger.info('=' * 60)
logger.info('AudioMOS 前后端一体模式启动')
logger.info('=' * 60)
logger.info(f'监听地址: {host}:{port}')

uvicorn.run(
    'app.main:app',
    host=host,
    port=port,
    reload=False,
    access_log=True
)
PYEOF
    
    nohup python "$SCRIPT_DIR/.start_server.py" > "$SCRIPT_DIR/logs/unified.log" 2>&1 &
    
    UNIFIED_PID=$!
    echo "$UNIFIED_PID" > "$SCRIPT_DIR/.unified.pid"
    
    echo "服务已启动, PID: $UNIFIED_PID"
    
    # 等待服务就绪
    echo ""
    echo "等待服务就绪..."
    local check_count=0
    local max_wait=30
    
    while [ $check_count -lt $max_wait ]; do
        if ! ps -p "$UNIFIED_PID" > /dev/null 2>&1; then
            echo "❌ 服务进程已退出，启动失败"
            echo "查看日志: tail -n 50 $SCRIPT_DIR/logs/unified.log"
            rm -f "$SCRIPT_DIR/.unified.pid"
            return 1
        fi
        
    # 健康检查: 如果 host 是 0.0.0.0, 用 127.0.0.1 连接
        local check_host="$unified_host"
        if [ "$check_host" = "0.0.0.0" ]; then
            check_host="127.0.0.1"
        fi
        
        if curl -s "http://$check_host:$unified_port/health" > /dev/null 2>&1; then
            echo ""
            echo "================================"
            echo "  ✅ AudioMOS 启动成功!"
            echo "================================"
            echo ""
            echo "🌐 访问地址: http://$unified_host:$unified_port"
            echo "📚 API文档:  http://$unified_host:$unified_port/docs"
            echo ""
            echo "默认登录账号:"
            echo "  用户名: admin"
            echo "  密码:   tp123456"
            echo ""
            return 0
        fi
        
        sleep 1
        check_count=$((check_count + 1))
        printf "\r  检查中... %d/%d 秒" "$check_count" "$max_wait"
    done
    
    echo ""
    echo "⚠️  服务启动超时，可能仍在初始化中"
    echo "查看日志: tail -f $SCRIPT_DIR/logs/unified.log"
    return 0
}

# 停止一体服务
stop_unified() {
    local pid_stopped=false

    if [ -f "$SCRIPT_DIR/.unified.pid" ]; then
        local pid=$(cat "$SCRIPT_DIR/.unified.pid")
        if ps -p "$pid" > /dev/null 2>&1; then
            echo "停止前后端一体服务 (PID: $pid)..."
            # 先尝试温和终止
            kill "$pid" 2>/dev/null
            # 等待进程结束，最多等待5秒
            local count=0
            while ps -p "$pid" > /dev/null 2>&1 && [ $count -lt 5 ]; do
                sleep 1
                count=$((count + 1))
            done
            # 如果还在运行，强制终止
            if ps -p "$pid" > /dev/null 2>&1; then
                echo "  进程未响应，强制终止..."
                kill -9 "$pid" 2>/dev/null
                sleep 1
            fi
            # 验证进程是否已终止
            if ps -p "$pid" > /dev/null 2>&1; then
                echo "⚠️  警告: 进程 $pid 可能仍在运行"
            else
                echo "✅ 服务已停止"
                pid_stopped=true
            fi
        else
            pid_stopped=true
        fi
        rm -f "$SCRIPT_DIR/.unified.pid"
    fi

    # 额外检查：查找并停止所有项目相关的 Python 一体服务进程
    if [ "$pid_stopped" = false ]; then
        local python_pids=$(pgrep -f "uvicorn.*app.main:app" | while read pid; do
            if pwdx "$pid" 2>/dev/null | grep -q "$SCRIPT_DIR/backend"; then
                echo "$pid"
            fi
        done)

        if [ -n "$python_pids" ]; then
            echo "发现残留的一体服务进程，正在停止..."
            for pid in $python_pids; do
                echo "  停止 Python 进程 (PID: $pid)"
                kill -9 "$pid" 2>/dev/null || true
            done
        fi
    fi

    # 最后手段: 按端口清理（仅 LISTEN 状态的进程）
    local port="${AUDIOMOS_PORT:-$(read_yaml_value "$CONFIG_FILE" "port" "8002")}"
    local port_pids=$(lsof -ti :"$port" -sTCP:LISTEN 2>/dev/null)
    if [ -n "$port_pids" ]; then
        echo "清理端口 $port 残留进程..."
        for pid in $port_pids; do
            local pname=$(ps -p "$pid" -o comm= 2>/dev/null)
            echo "  清理 (PID: $pid, $pname)"
            kill -9 "$pid" 2>/dev/null || true
        done
        sleep 1
        if lsof -ti :"$port" -sTCP:LISTEN > /dev/null 2>&1; then
            echo "⚠️  警告: 端口 $port 仍有进程残留，请手动检查"
        else
            echo "✅ 端口 $port 已释放"
        fi
    fi
}

# 重启服务
restart_services() {
    echo "================================"
    echo "  重启 AudioMOS 服务"
    echo "================================"
    echo ""
    stop_services
    stop_unified
    sleep 2
    start_services
}

# 创建日志目录
mkdir -p "$SCRIPT_DIR/logs"

# 主逻辑
case "${1:-start}" in
    start)
        shift
        start_unified "$@"
        ;;
    stop)
        stop_services
        stop_unified
        # 最后手段：直接清理占用配置端口的进程（仅 LISTEN 状态）
        _port="${AUDIOMOS_PORT:-$(read_yaml_value "$CONFIG_FILE" "port" "8002")}"
        _port_pids=$(lsof -ti :"$_port" -sTCP:LISTEN 2>/dev/null)
        if [ -n "$_port_pids" ]; then
            echo "发现端口 $_port 仍有进程占用，强制清理..."
            for _pid in $_port_pids; do
                _pname=$(ps -p "$_pid" -o comm= 2>/dev/null)
                echo "  清理进程 (PID: $_pid, $_pname)"
                kill -9 "$_pid" 2>/dev/null || true
            done
            sleep 1
            if lsof -ti :"$_port" -sTCP:LISTEN > /dev/null 2>&1; then
                echo "⚠️  警告: 端口 $_port 仍有进程残留"
            else
                echo "✅ 端口 $_port 已释放"
            fi
        fi
        unset _port _port_pids _pid _pname
        echo ""
        echo "================================"
        echo "  服务已停止"
        echo "================================"
        echo ""
        ;;
    restart)
        stop_services
        stop_unified
        sleep 2
        shift
        start_unified "$@"
        ;;
    status)
        show_status
        # 检查服务状态
        if [ -f "$SCRIPT_DIR/.unified.pid" ]; then
            _pid=$(cat "$SCRIPT_DIR/.unified.pid")
            if ps -p "$_pid" > /dev/null 2>&1; then
                echo ""
                echo "前后端一体服务:"
                echo "  ✅ 运行中 (PID: $_pid)"
            fi
        fi
        unset _pid
        ;;
    models)
        check_models
        ;;
    help|--help|-h)
        show_help
        ;;
    *)
        echo "未知命令: $1"
        echo ""
        show_help
        exit 1
        ;;
esac

