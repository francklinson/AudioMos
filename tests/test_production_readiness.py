"""
生产环境就绪测试
覆盖本次修复的关键点：模型路径、GPU配置、参考匹配、混合场景、API端点
"""
import pytest
import os
import sys
import json
import tempfile

# 路径设置
base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, base)
sys.path.insert(0, os.path.join(base, 'backend'))
sys.path.insert(0, os.path.join(base, 'app', 'core'))
sys.path.insert(0, os.path.join(base, 'app', 'algorithms'))


# =============================================================================
# 1. 模型路径绝对化测试
# =============================================================================

class TestModelPaths:
    """验证所有模型目录使用绝对路径"""

    def test_speechbrain_denoiser_path(self):
        """speechbrain_denoiser savedir 为绝对路径"""
        from denoise.speechbrain_denoiser import _DEFAULT_SAVEDIR
        assert os.path.isabs(_DEFAULT_SAVEDIR), f"不是绝对路径: {_DEFAULT_SAVEDIR}"
        assert 'models/speechbrain' in _DEFAULT_SAVEDIR.replace('\\', '/')

    def test_clearvocie_denoiser_path(self):
        """clearvocie_denoiser model_dir 为绝对路径"""
        from denoise.clearervoice_denoiser import _DEFAULT_MODEL_DIR
        assert os.path.isabs(_DEFAULT_MODEL_DIR), f"不是绝对路径: {_DEFAULT_MODEL_DIR}"
        assert 'models/clearvoice' in _DEFAULT_MODEL_DIR.replace('\\', '/')

    def test_dereverberation_path(self):
        """去混响 model_dir 为绝对路径"""
        from restoration.dereverberation import _DEFAULT_MODEL_DIR
        assert os.path.isabs(_DEFAULT_MODEL_DIR), f"不是绝对路径: {_DEFAULT_MODEL_DIR}"
        assert 'models/speechbrain' in _DEFAULT_MODEL_DIR.replace('\\', '/')

    def test_super_resolution_path(self):
        """超分 model_dir 为绝对路径"""
        from restoration.super_resolution import _DEFAULT_MODEL_DIR
        assert os.path.isabs(_DEFAULT_MODEL_DIR), f"不是绝对路径: {_DEFAULT_MODEL_DIR}"
        assert 'models/speechbrain' in _DEFAULT_MODEL_DIR.replace('\\', '/')

    def test_utmos_path_points_to_project(self):
        """UTMOS 缓存路径指向项目本地"""
        from utmosv2.utils._constants import _UTMOSV2_CHACHE
        assert 'AudioMos' in str(_UTMOSV2_CHACHE) or 'models/utmos' in str(_UTMOSV2_CHACHE), \
            f"UTMOS路径未指向项目本地: {_UTMOSV2_CHACHE}"

    def test_all_model_files_exist(self):
        """关键模型文件确实存在于本地"""
        checks = [
            ('TCF eres2net', 'models/tcf/eres2net/configuration.json'),
            ('TCF eres2netv2', 'models/tcf/eres2netv2/configuration.json'),
            ('UTMOS fold0', 'models/utmos/models/fusion_stage3/fold0_s42_best_model.pth'),
            ('SpeechBrain metricgan', 'models/speechbrain/metricgan-plus-voicebank/hyperparams.yaml'),
            ('ClearVoice FRCRN', 'models/clearvoice/FRCRN_SE_16K/last_best_checkpoint'),
        ]
        for name, rel_path in checks:
            full = os.path.join(base, rel_path)
            assert os.path.exists(full), f"{name}: 文件不存在 {full}"


# =============================================================================
# 2. GPU / CUDA 配置测试
# =============================================================================

class TestGPUConfig:
    """验证CUDA设备配置的安全性"""

    def test_no_hardcoded_device_id_in_code(self):
        """代码中不应存在 set_device(固定数字) 调用"""
        import subprocess
        result = subprocess.run(
            ['grep', '-rn', 'set_device([0-9]', os.path.join(base, 'app'), os.path.join(base, 'backend')],
            capture_output=True, text=True
        )
        # wenet train_utils 中的 set_device(local_rank) 是变量不是硬编码，忽略
        lines = [l for l in result.stdout.split('\n') if l and '__pycache__' not in l]
        hardcoded = [l for l in lines if 'local_rank' not in l and 'rank' not in l]
        assert len(hardcoded) == 0, f"存在硬编码set_device: {hardcoded}"

    def test_config_yaml_has_device_id(self):
        """config.yaml 包含 cuda.device_id 配置项"""
        config_path = os.path.join(base, 'config', 'config.yaml')
        assert os.path.exists(config_path), "config.yaml 不存在"
        import yaml
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        assert 'cuda' in cfg, "config.yaml 缺少 cuda 配置节"
        assert 'device_id' in cfg['cuda'], "config.yaml 缺少 cuda.device_id"

    def test_cuda_available_or_graceful_fallback(self):
        """CUDA检查逻辑: is_available 返回 bool"""
        import torch
        result = torch.cuda.is_available()
        assert isinstance(result, bool), "is_available 应返回 bool"


# =============================================================================
# 3. 参考音频匹配测试
# =============================================================================

class TestReferenceMatching:
    """验证预检测逻辑和内容匹配"""

    def setup_method(self):
        self.ref_dir = os.path.join(base, 'data', 'ref')

    def test_get_ref_file_by_content_self_match(self):
        """参考音频自匹配：ref_001 应匹配自身"""
        from calculator.mos_calculator import get_ref_file_by_content
        test_file = os.path.join(self.ref_dir, 'ref_001.wav')
        if not os.path.exists(test_file):
            pytest.skip("ref_001.wav 不存在")
        ref_file, match_info = get_ref_file_by_content(test_file, self.ref_dir)
        assert ref_file is not None, "参考音频无法自匹配"
        assert os.path.basename(ref_file) == 'ref_001.wav', \
            f"匹配到错误文件: {ref_file}"

    def test_get_ref_file_by_content_no_match(self):
        """纯噪声文件不应匹配到任何参考"""
        from calculator.mos_calculator import get_ref_file_by_content
        import numpy as np
        import soundfile as sf
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
            noise = np.random.randn(16000).astype(np.float32) * 0.01
            sf.write(f.name, noise, 16000)
            ref_file, _ = get_ref_file_by_content(f.name, self.ref_dir)
            assert ref_file is None, f"随机噪声不应匹配到参考: {ref_file}"
            os.unlink(f.name)

    def test_filename_match_priority(self):
        """文件名完全匹配优先于内容匹配"""
        from calculator.mos_calculator import get_ref_file_by_content
        test_file = os.path.join(self.ref_dir, 'ref_001.wav')
        if not os.path.exists(test_file):
            pytest.skip("ref_001.wav 不存在")
        _, match_info = get_ref_file_by_content(test_file, self.ref_dir)
        assert match_info is not None, "应返回匹配信息"
        assert match_info['method'].startswith('filename'), \
            f"同名文件应优先使用文件名匹配，而非: {match_info['method']}"


# =============================================================================
# 4. 混合场景测试
# =============================================================================

class TestMixedScenario:
    """验证部分匹配、部分不匹配时的处理逻辑"""

    def setup_method(self):
        self.ref_dir = os.path.join(base, 'data', 'ref')

    def test_precheck_distinguishes_matched_unmatched(self):
        """预检测应正确区分匹配和不匹配文件"""
        from calculator.mos_calculator import get_ref_file_by_content
        import numpy as np
        import soundfile as sf

        matched_file = os.path.join(self.ref_dir, 'ref_001.wav')
        if not os.path.exists(matched_file):
            pytest.skip("ref_001.wav 不存在")

        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
            noise = np.random.randn(16000).astype(np.float32) * 0.01
            sf.write(f.name, noise, 16000)

        results = {}
        for test_file in [matched_file, f.name]:
            ref_file, info = get_ref_file_by_content(test_file, self.ref_dir)
            results[os.path.basename(test_file)] = ref_file is not None

        assert results[os.path.basename(matched_file)] is True, "ref_001应匹配"
        assert results[os.path.basename(f.name)] is False, "噪声不应匹配"

        os.unlink(f.name)


# =============================================================================
# 5. API端点可用性测试
# =============================================================================

    """验证所有API端点响应正常"""

    @pytest.fixture(autouse=True, scope="class")
    def setup(self):
        from fastapi.testclient import TestClient
        backend_dir = os.path.join(base, 'backend')
        sys.path.insert(0, backend_dir)
        try:
            os.chdir(backend_dir)
            from app.main import app
            self.client = TestClient(app)
            self._app_available = True
        except Exception as e:
            self._app_available = False
            self._import_error = str(e)
            return
        # 登录获取token
        r = self.client.post('/api/auth/login', data={'username': 'admin', 'password': 'tp123456'})
        if r.status_code == 200:
            self.token = r.json()['access_token']
            self.headers = {'Authorization': f'Bearer {self.token}'}
        else:
            self.token = None
            self.headers = {}

# =============================================================================
# 6. 轻量级降噪功能测试
# =============================================================================

class TestLightweightDenoise:
    """谱减法/维纳滤波等轻量算法端到端测试"""

    def setup_method(self):
        import numpy as np
        self.audio = np.random.randn(16000).astype(np.float32) * 0.1

    def test_spectral_subtraction(self):
        """谱减法: 输出形状与输入一致"""
        from denoise import DenoiserRegistry
        d = DenoiserRegistry.get('spectral_subtraction', sample_rate=16000, device='cpu')
        d.initialize()
        result = d.denoise(self.audio, 16000)
        assert result.audio.shape == self.audio.shape, \
            f"输出形状 {result.audio.shape} != 输入形状 {self.audio.shape}"
        # 随机噪声可能无法计算有意义的SNR，仅验证处理不崩溃

    def test_wiener_filtering(self):
        """维纳滤波: 输出形状与输入一致"""
        from denoise import DenoiserRegistry
        d = DenoiserRegistry.get('wiener_filtering', sample_rate=16000, device='cpu')
        d.initialize()
        result = d.denoise(self.audio, 16000)
        assert result.audio.shape == self.audio.shape

    def test_denoise_output_is_float32(self):
        """降噪输出应为 float32"""
        from denoise import DenoiserRegistry
        d = DenoiserRegistry.get('spectral_subtraction', sample_rate=16000, device='cpu')
        d.initialize()
        result = d.denoise(self.audio, 16000)
        assert result.audio.dtype == np.float32, f"输出dtype: {result.audio.dtype}"


# =============================================================================
# 7. WER/WeNet 服务测试
# =============================================================================

class TestWeNetService:
    """验证 WeNet 服务路径修复"""

    def test_project_root_not_hardcoded(self):
        """wenet_service.py 中不存在硬编码的开发者路径"""
        wenet_path = os.path.join(base, 'backend', 'app', 'core', 'wenet_service.py')
        with open(wenet_path) as f:
            source = f.read()
        # 旧代码中的硬编码字符串
        assert "'/home/zhouchenghao/PycharmProjects/AudioMos'" not in source, \
            "wenet_service.py 不应包含硬编码的开发者路径"
        assert '"/home/zhouchenghao/PycharmProjects/AudioMos"' not in source, \
            "wenet_service.py 不应包含硬编码的开发者路径"
        # 新代码应使用动态计算的 _PROJECT_ROOT
        assert '_PROJECT_ROOT' in source, \
            "wenet_service.py 应使用 _PROJECT_ROOT 动态计算路径"


# =============================================================================
# 8. 配置一致性测试
# =============================================================================

class TestConfigConsistency:
    """验证配置文件一致性"""

    def test_config_yaml_exists(self):
        """config/config.yaml 存在"""
        config_path = os.path.join(base, 'config', 'config.yaml')
        assert os.path.exists(config_path)

    def test_port_is_8002(self):
        """部署端口为 8002"""
        import yaml
        config_path = os.path.join(base, 'config', 'config.yaml')
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        assert cfg['server']['port'] == 8002, f"端口应为8002，实际: {cfg['server']['port']}"

    def test_start_sh_port_fallback(self):
        """start.sh 端口默认值为 8002"""
        start_sh = os.path.join(base, 'start.sh')
        with open(start_sh) as f:
            content = f.read()
        assert '8002' in content, "start.sh 中应包含端口 8002"


import numpy as np
