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
BACKEND_HOST="${AUDIOMOS_BACKEND_HOST:-$(read_nested_yaml "$CONFIG_FILE" "server" "backend" "host" "0.0.0.0")}"
BACKEND_PORT="${AUDIOMOS_BACKEND_PORT:-$(read_nested_yaml "$CONFIG_FILE" "server" "backend" "port" "8000")}"
FRONTEND_HOST="${AUDIOMOS_FRONTEND_HOST:-$(read_nested_yaml "$CONFIG_FILE" "server" "frontend" "host" "0.0.0.0")}"
FRONTEND_PORT="${AUDIOMOS_FRONTEND_PORT:-$(read_nested_yaml "$CONFIG_FILE" "server" "frontend" "port" "3000")}"

# 向后兼容旧的环境变量
if [ -n "$AUDIOMOS_HOST" ]; then
    BACKEND_HOST="$AUDIOMOS_HOST"
fi
if [ -n "$AUDIOMOS_PORT" ]; then
    BACKEND_PORT="$AUDIOMOS_PORT"
fi

# 显示配置信息
echo ""
echo "================================"
echo "  服务配置"
echo "================================"
echo ""
echo "  后端: $BACKEND_HOST:$BACKEND_PORT"
echo "  前端: $FRONTEND_HOST:$FRONTEND_PORT"
echo ""

# 如果配置的是 auto，进行自动检测
if [ "$BACKEND_HOST" = "auto" ]; then
    BACKEND_HOST=$(validate_host "$BACKEND_HOST" "后端服务")
fi
if [ "$FRONTEND_HOST" = "auto" ]; then
    FRONTEND_HOST=$(validate_host "$FRONTEND_HOST" "前端服务")
fi

echo ""

# PID 文件路径
BACKEND_PID_FILE="$SCRIPT_DIR/.backend.pid"
FRONTEND_PID_FILE="$SCRIPT_DIR/.frontend.pid"

# 显示帮助信息
show_help() {
    echo "================================"
    echo "  AudioMOS 服务管理脚本"
    echo "================================"
    echo ""
    echo "用法: ./start.sh [命令] [选项]"
    echo ""
    echo "命令:"
    echo "  start           启动服务（默认前后端一体模式）"
    echo "  dev             启动前后端分离模式（适合开发）"
    echo "  stop            停止所有服务"
    echo "  restart         重启服务"
    echo "  status          查看服务状态"
    echo "  models          检查模型文件状态"
    echo "  build-frontend  构建前端静态文件"
    echo "  help            显示帮助信息"
    echo ""
    echo "选项:"
    echo "  --port <port>   指定端口"
    echo "  --host <host>   指定地址"
    echo ""
    echo "模式说明:"
    echo "  start - 前后端一体模式（单服务单端口，推荐部署）"
    echo "  dev   - 前后端分离模式（2个端口，适合开发）"
    echo ""
    echo "当前配置:"
    echo "  后端: $BACKEND_HOST:$BACKEND_PORT"
    echo "  前端: $FRONTEND_HOST:$FRONTEND_PORT"
    echo ""
    echo "环境变量:"
    echo "  AUDIOMOS_BACKEND_HOST    后端主机地址"
    echo "  AUDIOMOS_BACKEND_PORT    后端端口"
    echo "  AUDIOMOS_FRONTEND_HOST   前端主机地址"
    echo "  AUDIOMOS_FRONTEND_PORT   前端端口"
    echo ""
}

# 检查服务是否正在运行
check_status() {
    local backend_running=false
    local frontend_running=false

    if [ -f "$BACKEND_PID_FILE" ]; then
        local backend_pid=$(cat "$BACKEND_PID_FILE")
        if ps -p "$backend_pid" > /dev/null 2>&1; then
            backend_running=true
        fi
    fi

    if [ -f "$FRONTEND_PID_FILE" ]; then
        local frontend_pid=$(cat "$FRONTEND_PID_FILE")
        if ps -p "$frontend_pid" > /dev/null 2>&1; then
            frontend_running=true
        fi
    fi

    if $backend_running && $frontend_running; then
        echo "running"
    elif $backend_running; then
        echo "backend_only"
    elif $frontend_running; then
        echo "frontend_only"
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
        echo "✅ 后端服务: 运行中 (PID: $(cat $BACKEND_PID_FILE))"
        echo "   地址: http://$BACKEND_HOST:$BACKEND_PORT"
        echo ""
        echo "✅ 前端服务: 运行中 (PID: $(cat $FRONTEND_PID_FILE))"
        echo "   地址: http://$FRONTEND_HOST:$FRONTEND_PORT"
        echo ""
        echo "🌐 前端访问: http://$FRONTEND_HOST:$FRONTEND_PORT"
        echo "📡 后端API:  http://$BACKEND_HOST:$BACKEND_PORT"
        echo "📚 API文档:  http://$BACKEND_HOST:$BACKEND_PORT/docs"
    elif [ "$status" = "backend_only" ]; then
        echo "✅ 后端服务: 运行中 (PID: $(cat $BACKEND_PID_FILE))"
        echo "   地址: http://$BACKEND_HOST:$BACKEND_PORT"
        echo ""
        echo "❌ 前端服务: 未运行"
    elif [ "$status" = "frontend_only" ]; then
        echo "❌ 后端服务: 未运行"
        echo ""
        echo "✅ 前端服务: 运行中 (PID: $(cat $FRONTEND_PID_FILE))"
        echo "   地址: http://$FRONTEND_HOST:$FRONTEND_PORT"
    else
        echo "❌ 后端服务: 未运行"
        echo "❌ 前端服务: 未运行"
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
        local utmos_count=$(find "$utmos_model_dir" -name "*.pth" | wc -l)
        echo "   模型文件:"
        for f in "$utmos_model_dir"/fold*.pth; do
            if [ -f "$f" ]; then
                local fname=$(basename "$f")
                local fsize=$(du -sh "$f" 2>/dev/null | cut -f1)
                echo "      ✅ $fname ($fsize)"
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

    # 汇总
    echo "================================"
    if $all_ready; then
        echo "  ✅ 所有核心模型已就绪"
    else
        echo "  ⚠️  部分模型缺失"
        echo ""
        echo "请运行以下命令下载缺失的模型:"
        echo "  python download_models.py"
    fi
    echo "================================"
    echo ""

    return $([ "$all_ready" = true ] && echo 0 || echo 1)
}

# 启动服务
start_services() {
    local status=$(check_status)

    if [ "$status" = "running" ]; then
        echo "服务已经在运行中!"
        show_status
        return 0
    fi

    echo "================================"
    echo "  AudioMOS 项目启动脚本"
    echo "================================"
    echo ""
    echo "项目路径: $SCRIPT_DIR"
    echo ""
    echo "当前配置:"
    echo "  后端: $BACKEND_HOST:$BACKEND_PORT"
    echo "  前端: $FRONTEND_HOST:$FRONTEND_PORT"
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

    # 安装缺少的pesq(如果尚未安装)
    if ! pip show pesq > /dev/null 2>&1; then
        echo "安装pesq依赖..."
        pip install pesq==0.0.1 2>/dev/null || echo "pesq安装失败(可选依赖)"
    fi

    # 检查模型文件
    echo ""
    check_models
    local models_status=$?

    if [ $models_status -ne 0 ]; then
        echo "⚠️  模型检查未通过,是否继续启动? (y/N)"
        read -r response
        if [[ ! "$response" =~ ^[Yy]$ ]]; then
            echo "启动已取消"
            echo "请运行 'python download_models.py' 下载缺失的模型"
            return 1
        fi
        echo "继续启动(部分功能可能不可用)..."
        echo ""
    fi

    # 确保在项目根目录
    cd "$SCRIPT_DIR"

    # 启动后端
    if [ "$status" = "stopped" ] || [ "$status" = "frontend_only" ]; then
        echo ""
        echo "启动后端服务..."
        echo "  地址: http://$BACKEND_HOST:$BACKEND_PORT"
        cd backend
        # 设置环境变量供后端使用
        export AUDIOMOS_BACKEND_HOST="$BACKEND_HOST"
        export AUDIOMOS_BACKEND_PORT="$BACKEND_PORT"
        nohup python run.py > "$SCRIPT_DIR/logs/backend.log" 2>&1 &
        BACKEND_PID=$!
        echo "$BACKEND_PID" > "$BACKEND_PID_FILE"
        echo "后端服务已启动, PID: $BACKEND_PID"
        
        # 等待后端服务真正启动
        echo ""
        echo "等待后端服务就绪..."
        local backend_ready=false
        local backend_check_count=0
        local backend_max_wait=30
        
        while [ $backend_check_count -lt $backend_max_wait ]; do
            # 检查进程是否还在运行
            if ! ps -p "$BACKEND_PID" > /dev/null 2>&1; then
                echo "❌ 后端进程已退出，启动失败"
                echo "查看日志: tail -n 50 $SCRIPT_DIR/logs/backend.log"
                rm -f "$BACKEND_PID_FILE"
                return 1
            fi
            
            # 检查端口是否监听
            if lsof -i :"$BACKEND_PORT" > /dev/null 2>&1; then
                # 尝试访问健康检查接口
                if curl -s "http://$BACKEND_HOST:$BACKEND_PORT/health" > /dev/null 2>&1; then
                    echo "✅ 后端服务已就绪"
                    backend_ready=true
                    break
                fi
            fi
            
            sleep 1
            backend_check_count=$((backend_check_count + 1))
            printf "\r  检查中... %d/%d 秒" "$backend_check_count" "$backend_max_wait"
        done
        
        echo ""
        
        if [ "$backend_ready" = false ]; then
            echo "⚠️  后端服务启动超时，可能仍在初始化中"
            echo "查看日志: tail -f $SCRIPT_DIR/logs/backend.log"
        fi
    fi

    # 确保回到项目根目录
    cd "$SCRIPT_DIR"

    # 检查前端依赖
    if [ ! -d "frontend/node_modules" ]; then
        echo ""
        echo "正在安装前端依赖..."
        cd frontend
        npm install
        cd "$SCRIPT_DIR"
    fi

    # 启动前端
    local frontend_ready=false
    if [ "$status" = "stopped" ] || [ "$status" = "backend_only" ]; then
        echo ""
        echo "启动前端服务..."
        echo "  地址: http://$FRONTEND_HOST:$FRONTEND_PORT"
        cd frontend
        # 设置环境变量供 Vite 配置读取
        export AUDIOMOS_FRONTEND_HOST="$FRONTEND_HOST"
        export AUDIOMOS_FRONTEND_PORT="$FRONTEND_PORT"
        # 不再传递 --host 参数，让 Vite 使用配置文件中的设置
        # Vite 配置会将 0.0.0.0 转换为 true，避免 Node.js 解析问题
        nohup npx vite > "$SCRIPT_DIR/logs/frontend.log" 2>&1 &
        FRONTEND_PID=$!
        echo "$FRONTEND_PID" > "$FRONTEND_PID_FILE"
        echo "前端服务已启动, PID: $FRONTEND_PID"
        
        # 等待前端服务真正启动
        echo ""
        echo "等待前端服务就绪..."
        local frontend_check_count=0
        local frontend_max_wait=30
        
        while [ $frontend_check_count -lt $frontend_max_wait ]; do
            # 检查进程是否还在运行
            if ! ps -p "$FRONTEND_PID" > /dev/null 2>&1; then
                echo "❌ 前端进程已退出，启动失败"
                echo "查看日志: tail -n 50 $SCRIPT_DIR/logs/frontend.log"
                rm -f "$FRONTEND_PID_FILE"
                return 1
            fi
            
            # 检查端口是否监听
            if lsof -i :"$FRONTEND_PORT" > /dev/null 2>&1; then
                # 尝试访问前端页面
                if curl -s "http://$FRONTEND_HOST:$FRONTEND_PORT" > /dev/null 2>&1; then
                    echo "✅ 前端服务已就绪"
                    frontend_ready=true
                    break
                fi
            fi
            
            sleep 1
            frontend_check_count=$((frontend_check_count + 1))
            printf "\r  检查中... %d/%d 秒" "$frontend_check_count" "$frontend_max_wait"
        done
        
        echo ""
        
        if [ "$frontend_ready" = false ]; then
            echo "⚠️  前端服务启动超时，可能仍在初始化中"
            echo "查看日志: tail -f $SCRIPT_DIR/logs/frontend.log"
        fi
    fi

    echo ""
    echo "================================"
    
    # 最终状态检查
    if [ "$backend_ready" = true ] && [ "$frontend_ready" = true ]; then
        echo "  ✅ AudioMOS 启动成功!"
    elif [ "$backend_ready" = true ]; then
        echo "  ⚠️  AudioMOS 部分启动 (后端正常，前端可能仍在初始化)"
    elif [ "$frontend_ready" = true ]; then
        echo "  ⚠️  AudioMOS 部分启动 (前端正常，后端可能仍在初始化)"
    else
        echo "  ⚠️  AudioMOS 启动状态未知，请检查日志"
    fi
    echo "================================"
    echo ""
    echo "🌐 前端访问: http://$FRONTEND_HOST:$FRONTEND_PORT"
    echo "📡 后端API:  http://$BACKEND_HOST:$BACKEND_PORT"
    echo "📚 API文档:  http://$BACKEND_HOST:$BACKEND_PORT/docs"
    echo ""
    echo "默认登录账号:"
    echo "  用户名: admin"
    echo "  密码:   tp123456"
    echo ""
    echo "使用 './start.sh stop' 停止服务"
    echo ""
}

# 停止服务
stop_services() {
    echo "================================"
    echo "  停止 AudioMOS 服务"
    echo "================================"
    echo ""

    local backend_stopped=false
    local frontend_stopped=false

    # 停止后端
    if [ -f "$BACKEND_PID_FILE" ]; then
        local backend_pid=$(cat "$BACKEND_PID_FILE")
        if ps -p "$backend_pid" > /dev/null 2>&1; then
            echo "正在停止后端服务 (PID: $backend_pid)..."
            kill "$backend_pid" 2>/dev/null
            sleep 2
            # 强制终止如果还在运行
            if ps -p "$backend_pid" > /dev/null 2>&1; then
                kill -9 "$backend_pid" 2>/dev/null
            fi
            echo "✅ 后端服务已停止"
        else
            echo "后端服务未运行"
        fi
        rm -f "$BACKEND_PID_FILE"
        backend_stopped=true
    fi

    # 停止前端 - 不仅通过 PID 文件，还查找所有 Vite 进程
    if [ -f "$FRONTEND_PID_FILE" ]; then
        local frontend_pid=$(cat "$FRONTEND_PID_FILE")
        if ps -p "$frontend_pid" > /dev/null 2>&1; then
            echo "正在停止前端服务 (PID: $frontend_pid)..."
            kill "$frontend_pid" 2>/dev/null
            sleep 2
            # 强制终止如果还在运行
            if ps -p "$frontend_pid" > /dev/null 2>&1; then
                kill -9 "$frontend_pid" 2>/dev/null
            fi
            echo "✅ 前端服务已停止"
        else
            echo "前端服务未运行"
        fi
        rm -f "$FRONTEND_PID_FILE"
        frontend_stopped=true
    fi

    # 额外检查：查找并停止所有项目相关的 Vite 和 Python 进程
    echo ""
    echo "检查残留进程..."

    # 查找项目目录下的 Vite 进程
    local vite_pids=$(pgrep -f "vite" | while read pid; do
        if pwdx "$pid" 2>/dev/null | grep -q "$SCRIPT_DIR/frontend"; then
            echo "$pid"
        fi
    done)

    if [ -n "$vite_pids" ]; then
        echo "发现残留的 Vite 进程，正在停止..."
        for pid in $vite_pids; do
            echo "  停止 Vite 进程 (PID: $pid)"
            kill -9 "$pid" 2>/dev/null || true
        done
        frontend_stopped=true
    fi

    # 查找项目目录下的 Python 后端进程
    local python_pids=$(pgrep -f "python.*run\.py" | while read pid; do
        if pwdx "$pid" 2>/dev/null | grep -q "$SCRIPT_DIR/backend"; then
            echo "$pid"
        fi
    done)

    if [ -n "$python_pids" ]; then
        echo "发现残留的 Python 后端进程，正在停止..."
        for pid in $python_pids; do
            echo "  停止 Python 进程 (PID: $pid)"
            kill -9 "$pid" 2>/dev/null || true
        done
        backend_stopped=true
    fi

    if ! $backend_stopped && ! $frontend_stopped; then
        echo "没有运行中的服务"
    fi

    echo ""
    echo "================================"
    echo "  分离模式服务已停止"
    echo "================================"
    echo ""
}

# 等待端口释放
wait_for_ports() {
    echo ""
    echo "等待端口释放..."
    local ports=($BACKEND_PORT $FRONTEND_PORT)
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

# 构建前端静态文件
build_frontend() {
    echo "================================"
    echo "  构建前端静态文件"
    echo "================================"
    echo ""
    
    # 检查前端目录
    if [ ! -d "$SCRIPT_DIR/frontend" ]; then
        echo "❌ 前端目录不存在: $SCRIPT_DIR/frontend"
        return 1
    fi
    
    # 激活虚拟环境
    source "$SCRIPT_DIR/.venv/bin/activate"
    
    cd "$SCRIPT_DIR/frontend"
    
    # 检查 node_modules
    if [ ! -d "node_modules" ]; then
        echo "📦 安装前端依赖..."
        npm install
        if [ $? -ne 0 ]; then
            echo "❌ 安装依赖失败"
            return 1
        fi
    fi
    
    # 构建前端
    echo "🔨 构建前端..."
    npm run build:prod
    
    if [ $? -ne 0 ]; then
        echo "❌ 构建失败"
        return 1
    fi
    
    # 检查构建输出
    if [ -f "$SCRIPT_DIR/backend/static/index.html" ]; then
        echo "✅ 前端构建成功"
        echo "   输出目录: $SCRIPT_DIR/backend/static/"
        return 0
    else
        echo "❌ 构建输出不存在"
        return 1
    fi
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
    
    # 检查前端是否已构建
    if [ ! -f "$SCRIPT_DIR/backend/static/index.html" ]; then
        echo ""
        echo "⚠️  未检测到前端构建文件"
        echo ""
        
        # 检查 node_modules
        if [ ! -d "$SCRIPT_DIR/frontend/node_modules" ]; then
            echo "📦 安装前端依赖..."
            cd "$SCRIPT_DIR/frontend"
            npm install
            cd "$SCRIPT_DIR"
        fi
        
        echo "🔨 构建前端..."
        cd "$SCRIPT_DIR/frontend"
        npm run build:prod
        cd "$SCRIPT_DIR"
        
        if [ ! -f "$SCRIPT_DIR/backend/static/index.html" ]; then
            echo "❌ 前端构建失败"
            return 1
        fi
        echo "✅ 前端构建成功"
    else
        echo "✅ 检测到前端构建文件"
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
    
    # 停止已有服务
    stop_services > /dev/null 2>&1
    
    # 启动一体服务
    echo ""
    echo "启动服务..."
    export AUDIOMOS_BACKEND_HOST="$unified_host"
    export AUDIOMOS_BACKEND_PORT="$unified_port"
    
    cd "$SCRIPT_DIR/backend"
    nohup python -c "
import sys
sys.path.insert(0, '.')
import uvicorn
from app.core.logging_config import logger

logger.info('=' * 60)
logger.info('AudioMOS 前后端一体模式启动')
logger.info('=' * 60)
logger.info(f'监听地址: $unified_host:$unified_port')

uvicorn.run(
    'app.main:app',
    host='$unified_host',
    port=$unified_port,
    reload=False,
    access_log=True
)
" > "$SCRIPT_DIR/logs/unified.log" 2>&1 &
    
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
        
        if curl -s "http://$unified_host:$unified_port/health" > /dev/null 2>&1; then
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
        # 默认使用前后端一体模式（适合部署）
        shift
        start_unified "$@"
        ;;
    dev)
        # 使用前后端分离模式（适合开发）
        echo "================================"
        echo "  启动开发模式（前后端分离）"
        echo "================================"
        echo ""
        start_services
        ;;
    stop)
        stop_services
        stop_unified
        wait_for_ports
        echo ""
        echo "================================"
        echo "  服务已停止"
        echo "================================"
        echo ""
        ;;
    restart)
        stop_services
        stop_unified
        wait_for_ports
        sleep 2
        shift
        start_unified "$@"
        ;;
    status)
        show_status
        # 检查一体服务状态
        if [ -f "$SCRIPT_DIR/.unified.pid" ]; then
            local pid=$(cat "$SCRIPT_DIR/.unified.pid")
            if ps -p "$pid" > /dev/null 2>&1; then
                echo ""
                echo "前后端一体服务:"
                echo "  ✅ 运行中 (PID: $pid)"
            fi
        fi
        ;;
    models)
        check_models
        ;;
    build-frontend)
        build_frontend
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
