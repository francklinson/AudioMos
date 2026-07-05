#!/bin/bash
# AudioMOS 前端快速测试脚本
# 用于快速验证前端基础功能是否正常

echo "=========================================="
echo "   AudioMOS 前端功能快速测试"
echo "=========================================="
echo ""

# 1. 检查服务是否启动
echo "[1/6] 检查服务状态..."
if curl -s http://localhost:8000/health | grep -q "healthy"; then
    echo "✅ 后端服务正常运行"
else
    echo "❌ 后端服务未启动"
    echo "   请先运行: ./start.sh"
    exit 1
fi
echo ""

# 2. 测试页面加载
echo "[2/6] 测试前端页面..."
STATUS=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:8000/)
if [ "$STATUS" = "200" ]; then
    echo "✅ 前端页面加载成功 (HTTP $STATUS)"
else
    echo "❌ 前端页面加载失败 (HTTP $STATUS)"
fi
echo ""

# 3. 测试登录API
echo "[3/6] 测试登录API..."
TOKEN=$(curl -s -X POST http://localhost:8000/api/auth/login \
    -d "username=admin&password=tp123456" | \
    grep -o '"access_token":"[^"]*' | \
    sed 's/"access_token":"//')

if [ -n "$TOKEN" ]; then
    echo "✅ 登录API正常，获取到Token"
    echo "   Token前缀: ${TOKEN:0:20}..."
else
    echo "❌ 登录API异常，无法获取Token"
fi
echo ""

# 4. 测试静态资源
echo "[4/6] 测试静态资源..."
CSS_STATUS=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:8000/css/app.css)
JS_STATUS=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:8000/js/app.js)

if [ "$CSS_STATUS" = "200" ]; then
    echo "✅ CSS文件加载成功"
else
    echo "❌ CSS文件加载失败 (HTTP $CSS_STATUS)"
fi

if [ "$JS_STATUS" = "200" ]; then
    echo "✅ JS文件加载成功"
else
    echo "❌ JS文件加载失败 (HTTP $JS_STATUS)"
fi
echo ""

# 5. 测试核心API接口
echo "[5/6] 测试核心API接口..."

# 测试MOS任务列表
MOS_STATUS=$(curl -s -o /dev/null -w "%{http_code}" \
    -H "Authorization: Bearer $TOKEN" \
    http://localhost:8000/api/mos/tasks)

if [ "$MOS_STATUS" = "200" ]; then
    echo "✅ MOS API接口正常"
else
    echo "❌ MOS API接口异常 (HTTP $MOS_STATUS)"
fi

# 测试音频修复算法列表
REST_STATUS=$(curl -s -o /dev/null -w "%{http_code}" \
    -H "Authorization: Bearer $TOKEN" \
    http://localhost:8000/api/restoration/algorithms)

if [ "$REST_STATUS" = "200" ]; then
    echo "✅ 音频修复API接口正常"
else
    echo "❌ 音频修复API接口异常 (HTTP $REST_STATUS)"
fi

# 测试参考音频列表
REF_STATUS=$(curl -s -o /dev/null -w "%{http_code}" \
    -H "Authorization: Bearer $TOKEN" \
    http://localhost:8000/api/reference-audio/list)

if [ "$REF_STATUS" = "200" ]; then
    echo "✅ 参考音频API接口正常"
else
    echo "❌ 参考音频API接口异常 (HTTP $REF_STATUS)"
fi
echo ""

# 6. 测试文件上传
echo "[6/6] 测试文件上传功能..."
# 创建测试音频（1秒静音）
TEST_FILE="/tmp/test_audio_quick.wav"

# 使用Python生成测试WAV文件
python3 << 'EOF'
import wave
import struct
import os

sample_rate = 16000
duration = 1.0
num_samples = int(sample_rate * duration)

buffer = []
for i in range(num_samples):
    buffer.append(0)

with wave.open('/tmp/test_audio_quick.wav', 'wb') as wav_file:
    wav_file.setnchannels(1)
    wav_file.setsampwidth(2)
    wav_file.setframerate(sample_rate)
    wav_file.writeframes(struct.pack('<' + 'h' * num_samples, *buffer))

print("测试音频文件已生成")
EOF

if [ -f "$TEST_FILE" ]; then
    UPLOAD_RESULT=$(curl -s -X POST \
        -H "Authorization: Bearer $TOKEN" \
        -F "files=@$TEST_FILE" \
        http://localhost:8000/api/mos/upload)
    
    if echo "$UPLOAD_RESULT" | grep -q "task_id"; then
        echo "✅ 文件上传功能正常"
        TASK_ID=$(echo "$UPLOAD_RESULT" | grep -o '"task_id":"[^"]*' | sed 's/"task_id":"//')
        echo "   任务ID: $TASK_ID"
    else
        echo "❌ 文件上传功能异常"
        echo "   响应: $UPLOAD_RESULT"
    fi
    
    # 清理测试文件
    rm -f "$TEST_FILE"
else
    echo "⚠️  无法生成测试音频文件"
fi
echo ""

# 总结
echo "=========================================="
echo "   测试结果总结"
echo "=========================================="
echo ""
echo "✅ 所有基础功能测试完成"
echo ""
echo "建议下一步："
echo "1. 打开浏览器访问: http://localhost:8000"
echo "2. 使用账号 admin / tp123456 登录"
echo "3. 手动测试完整功能流程"
echo ""
echo "详细测试指南请查看:"
echo "docs/前端测试指南.md"
echo ""
echo "=========================================="

exit 0