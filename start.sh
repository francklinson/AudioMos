#!/bin/bash

# AudioMOS 服务管理脚本
# 2026-05-12

# 获取脚本所在目录的绝对路径
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
cd "$SCRIPT_DIR"

# PID 文件路径
BACKEND_PID_FILE="$SCRIPT_DIR/.backend.pid"
FRONTEND_PID_FILE="$SCRIPT_DIR/.frontend.pid"

# 显示帮助信息
show_help() {
    echo "================================"
    echo "  AudioMOS 服务管理脚本"
    echo "================================"
    echo ""
    echo "用法: ./start.sh [命令]"
    echo ""
    echo "命令:"
    echo "  start    启动前后端服务"
    echo "  stop     停止前后端服务"
    echo "  restart  重启前后端服务"
    echo "  status   查看服务状态"
    echo "  models   检查模型文件状态"
    echo "  help     显示帮助信息"
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
        echo "✅ 前端服务: 运行中 (PID: $(cat $FRONTEND_PID_FILE))"
        echo ""
        echo "🌐 前端访问: http://localhost:3000"
        echo "📡 后端API:  http://localhost:8000"
        echo "📚 API文档:  http://localhost:8000/docs"
    elif [ "$status" = "backend_only" ]; then
        echo "✅ 后端服务: 运行中 (PID: $(cat $BACKEND_PID_FILE))"
        echo "❌ 前端服务: 未运行"
    elif [ "$status" = "frontend_only" ]; then
        echo "❌ 后端服务: 未运行"
        echo "✅ 前端服务: 运行中 (PID: $(cat $FRONTEND_PID_FILE))"
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
    echo "   使用6模型加权评估 (eres2net/eres2netv2/campplus/ecapa-tdnn/res2net/resnet34)"
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
            echo "   ✅ $model_name (项目路径)"
            ((tcf_available++))
        elif [ -f "$cache_path" ]; then
            echo "   ✅ $model_name (本地缓存)"
            ((tcf_available++))
        else
            echo "   ❌ $model_name 缺失"
            echo "      期望: $SCRIPT_DIR/models/tcf/$model_name/"
        fi
    done
    
    echo ""
    if [ $tcf_available -eq 0 ]; then
        echo "   ⚠️  所有TCF模型都缺失"
        all_ready=false
    elif [ $tcf_available -lt $tcf_total ]; then
        echo "   ⚠️  TCF模型部分可用 ($tcf_available/$tcf_total)"
        echo "      将使用可用模型进行加权评估"
    else
        echo "   ✅ 所有TCF模型已就绪 ($tcf_available/$tcf_total)"
    fi
    echo ""

    # 2. 检查WeNet模型 (WER语音识别) - 优先检查项目路径
    echo "🔍 检查 WeNet (语音识别) 模型..."
    local wenet_project_path="$SCRIPT_DIR/models/wenet/final.pt"
    local wenet_cache_path="$HOME/.wenet/wenetspeech/final.pt"
    
    if [ -f "$wenet_project_path" ]; then
        echo "   ✅ WeNet模型已就绪 (项目路径)"
        echo "      位置: $SCRIPT_DIR/models/wenet/"
    elif [ -f "$wenet_cache_path" ]; then
        echo "   ✅ WeNet模型已就绪 (本地缓存)"
        echo "      位置: $HOME/.wenet/wenetspeech/"
    else
        echo "   ❌ WeNet模型缺失"
        echo "      期望路径: $SCRIPT_DIR/models/wenet/"
        all_ready=false
    fi
    echo ""

    # 3. 检查NISQA模型
    echo "🔍 检查 NISQA 模型..."
    local nisqa_model_path="$SCRIPT_DIR/models/nisqa/weights/nisqa.tar"
    local nisqa_model_path_old="$SCRIPT_DIR/app/algorithms/nisqa/weights/nisqa.tar"
    if [ -f "$nisqa_model_path" ]; then
        echo "   ✅ NISQA模型已就绪"
        echo "      位置: $nisqa_model_path"
    elif [ -f "$nisqa_model_path_old" ]; then
        echo "   ✅ NISQA模型已就绪"
        echo "      位置: $nisqa_model_path_old"
    else
        echo "   ⚠️  NISQA模型缺失 (可选依赖)"
        echo "      期望路径: $nisqa_model_path"
    fi
    echo ""

    # 4. 检查DNSMOS模型
    echo "🔍 检查 DNSMOS 模型..."
    local dnsmos_p808_path="$SCRIPT_DIR/models/dnsmos/DNSMOS/model_v8.onnx"
    local dnsmos_primary_path="$SCRIPT_DIR/models/dnsmos/pDNSMOS/sig_bak_ovr.onnx"
    local dnsmos_p808_path_old="$SCRIPT_DIR/app/algorithms/dnsmos/DNSMOS/model_v8.onnx"
    local dnsmos_primary_path_old="$SCRIPT_DIR/app/algorithms/dnsmos/pDNSMOS/sig_bak_ovr.onnx"
    local dnsmos_ok=true

    if [ ! -f "$dnsmos_p808_path" ] && [ ! -f "$dnsmos_p808_path_old" ]; then
        echo "   ⚠️  DNSMOS P808模型缺失 (可选依赖)"
        echo "      期望路径: $dnsmos_p808_path"
        dnsmos_ok=false
    fi

    if [ ! -f "$dnsmos_primary_path" ] && [ ! -f "$dnsmos_primary_path_old" ]; then
        echo "   ⚠️  DNSMOS Primary模型缺失 (可选依赖)"
        echo "      期望路径: $dnsmos_primary_path"
        dnsmos_ok=false
    fi

    if $dnsmos_ok; then
        echo "   ✅ DNSMOS模型已就绪"
    fi
    echo ""

    # 5. 检查Scoreq模型 (通过Python检查)
    echo "🔍 检查 Scoreq 模块..."
    if source .venv/bin/activate && python -c "import scoreq; print('OK')" 2>/dev/null | grep -q "OK"; then
        echo "   ✅ Scoreq模块已安装"
        local scoreq_path=$(source .venv/bin/activate && python -c "import scoreq; print(scoreq.__file__)" 2>/dev/null)
        echo "      位置: $scoreq_path"
    else
        echo "   ⚠️  Scoreq模块未安装 (可选依赖)"
    fi
    echo ""

    # 6. 检查speechmetrics (通过Python检查)
    echo "🔍 检查 SpeechMetrics 模块..."
    if source .venv/bin/activate && PYTHONPATH="$SCRIPT_DIR/app/algorithms:$PYTHONPATH" python -c "import speechmetrics; print('OK')" 2>/dev/null | grep -q "OK"; then
        echo "   ✅ SpeechMetrics模块已就绪"
    else
        echo "   ⚠️  SpeechMetrics模块未就绪 (可选依赖)"
    fi
    echo ""

    # 7. 检查UTMOS模型
    echo "🔍 检查 UTMOS 模型..."
    # 添加app/algorithms到Python路径
    if source .venv/bin/activate && PYTHONPATH="$SCRIPT_DIR/app/algorithms:$PYTHONPATH" python -c "import utmosv2; print('OK')" 2>/dev/null | grep -q "OK"; then
        echo "   ✅ UTMOS模块已安装"
        echo "      说明: UTokyo-SaruLab MOS预测系统 (VoiceMOS 2024第一名)"
    else
        echo "   ⚠️  UTMOS模块未安装 (可选依赖)"
        echo "      安装命令: pip install -e ./app/algorithms/utmos"
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
        cd backend
        nohup python run.py > "$SCRIPT_DIR/logs/backend.log" 2>&1 &
        BACKEND_PID=$!
        echo "$BACKEND_PID" > "$BACKEND_PID_FILE"
        echo "后端服务已启动, PID: $BACKEND_PID"
        sleep 3
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
    if [ "$status" = "stopped" ] || [ "$status" = "backend_only" ]; then
        echo ""
        echo "启动前端服务..."
        cd frontend
        nohup npm run dev > "$SCRIPT_DIR/logs/frontend.log" 2>&1 &
        FRONTEND_PID=$!
        echo "$FRONTEND_PID" > "$FRONTEND_PID_FILE"
        echo "前端服务已启动, PID: $FRONTEND_PID"
    fi

    echo ""
    echo "================================"
    echo "  AudioMOS 启动成功!"
    echo "================================"
    echo ""
    echo "🌐 前端访问: http://localhost:3000"
    echo "📡 后端API:  http://localhost:8000"
    echo "📚 API文档:  http://localhost:8000/docs"
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

    # 停止前端
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

    if ! $backend_stopped && ! $frontend_stopped; then
        echo "没有运行中的服务"
    fi

    echo ""
    echo "================================"
    echo "  服务已停止"
    echo "================================"
    echo ""
}

# 重启服务
restart_services() {
    echo "================================"
    echo "  重启 AudioMOS 服务"
    echo "================================"
    echo ""
    stop_services
    sleep 2
    start_services
}

# 创建日志目录
mkdir -p "$SCRIPT_DIR/logs"

# 主逻辑
case "${1:-start}" in
    start)
        start_services
        ;;
    stop)
        stop_services
        ;;
    restart)
        restart_services
        ;;
    status)
        show_status
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
