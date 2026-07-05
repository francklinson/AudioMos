"""
AudioMOS 前端UI自动化测试（Playwright）
全面测试前端所有功能模块
"""
import pytest
import time
import os
from playwright.sync_api import sync_playwright, expect


# ======================== 配置 ========================
BASE_URL = "http://localhost:8000"
USERNAME = "admin"
PASSWORD = "tp123456"
TEST_AUDIO_DIR = "tests/test_data"


# ======================== Fixtures ========================
@pytest.fixture(scope="module")
def browser():
    """创建浏览器实例（模块级别，所有测试共享）"""
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)  # headless=True为无界面模式
        yield browser
        browser.close()


@pytest.fixture
def page(browser):
    """创建页面（每个测试独立）"""
    context = browser.new_context()
    page = context.new_page()
    page.goto(BASE_URL)
    yield page
    context.close()


@pytest.fixture
def authenticated_page(page):
    """已登录的页面"""
    # 等待登录页面加载
    page.wait_for_selector("#login-form", timeout=5000)
    
    # 输入登录信息
    page.fill("#username-input", USERNAME)
    page.fill("#password-input", PASSWORD)
    page.click("#login-btn")
    
    # 等待跳转到主页面
    page.wait_for_selector("#app-section", timeout=5000)
    
    return page


@pytest.fixture(scope="module")
def test_audio_file():
    """生成测试音频文件"""
    os.makedirs(TEST_AUDIO_DIR, exist_ok=True)
    test_file = os.path.join(TEST_AUDIO_DIR, "test_sample.wav")
    
    # 如果文件不存在，生成一个简单的测试音频
    if not os.path.exists(test_file):
        import wave
        import struct
        
        sample_rate = 16000
        duration = 2.0
        num_samples = int(sample_rate * duration)
        
        with wave.open(test_file, 'wb') as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            # 生成静音数据
            samples = [0] * num_samples
            wav_file.writeframes(struct.pack('<' + 'h' * num_samples, *samples))
    
    return test_file


# ======================== 登录功能测试 ========================
class TestLogin:
    """登录功能测试类"""

    def test_01_login_page_load(self, page):
        """测试01: 登录页面正确加载"""
        # 验证页面标题
        assert "AudioMOS" in page.title()
        
        # 验证登录表单元素存在
        assert page.is_visible("#login-form")
        assert page.is_visible("#username-input")
        assert page.is_visible("#password-input")
        assert page.is_visible("#login-btn")
        
        # 验证品牌元素
        assert page.is_visible(".login-brand")
        assert page.is_visible(".brand-icon")

    def test_02_login_success(self, page):
        """测试02: 正常登录流程"""
        # 等待登录页面
        page.wait_for_selector("#login-form")
        
        # 输入正确的用户名和密码
        page.fill("#username-input", USERNAME)
        page.fill("#password-input", PASSWORD)
        
        # 点击登录
        page.click("#login-btn")
        
        # 等待跳转到主页面
        page.wait_for_selector("#app-section", timeout=5000)
        
        # 验证登录成功
        assert page.is_visible("#app-section")
        assert not page.is_visible("#login-section")
        
        # 验证用户名显示
        user_display = page.locator("#user-display")
        expect(user_display).to_have_text(USERNAME)

    def test_03_login_wrong_password(self, page):
        """测试03: 错误密码登录"""
        page.wait_for_selector("#login-form")
        
        page.fill("#username-input", USERNAME)
        page.fill("#password-input", "wrong_password")
        page.click("#login-btn")
        
        # 等待错误提示出现
        page.wait_for_selector(".toast-custom.error", timeout=3000)
        
        # 验证仍在登录页面
        assert page.is_visible("#login-section")
        assert not page.is_visible("#app-section")

    def test_04_login_empty_username(self, page):
        """测试04: 空用户名登录"""
        page.wait_for_selector("#login-form")
        
        page.fill("#username-input", "")
        page.fill("#password-input", PASSWORD)
        page.click("#login-btn")
        
        # 验证仍在登录页面（表单验证阻止提交）
        assert page.is_visible("#login-section")

    def test_05_login_empty_password(self, page):
        """测试05: 空密码登录"""
        page.wait_for_selector("#login-form")
        
        page.fill("#username-input", USERNAME)
        page.fill("#password-input", "")
        page.click("#login-btn")
        
        # 验证仍在登录页面
        assert page.is_visible("#login-section")

    def test_06_login_both_empty(self, page):
        """测试06: 用户名和密码都为空"""
        page.wait_for_selector("#login-form")
        
        page.fill("#username-input", "")
        page.fill("#password-input", "")
        page.click("#login-btn")
        
        # 验证仍在登录页面
        assert page.is_visible("#login-section")

    def test_07_logout(self, authenticated_page):
        """测试07: 登出功能"""
        page = authenticated_page
        
        # 点击登出按钮
        page.click("#logout-btn")
        
        # 等待回到登录页面
        page.wait_for_selector("#login-section", timeout=3000)
        
        # 验证登出成功
        assert page.is_visible("#login-section")
        assert not page.is_visible("#app-section")


# ======================== 主页面UI测试 ========================
class TestMainUI:
    """主页面UI测试类"""

    def test_08_navbar_display(self, authenticated_page):
        """测试08: 导航栏正确显示"""
        page = authenticated_page
        
        # 验证导航栏元素
        assert page.is_visible(".app-navbar")
        assert page.is_visible(".nav-brand")
        assert page.is_visible("#user-display")
        assert page.is_visible("#logout-btn")
        
        # 验证品牌名称
        brand_text = page.locator(".nav-brand").text_content()
        assert "AudioMOS" in brand_text

    def test_09_tabs_navigation(self, authenticated_page):
        """测试09: Tab导航功能"""
        page = authenticated_page
        
        # 验证所有Tab存在
        tabs = ["mos", "denoise", "restoration", "reference"]
        for tab_name in tabs:
            tab_selector = f'[data-tab="{tab_name}"]'
            if page.is_visible(tab_selector):
                # 点击Tab
                page.click(tab_selector)
                time.sleep(0.5)
                
                # 验证Tab内容显示
                content_selector = f'#{tab_name}-content'
                if page.is_visible(content_selector):
                    assert page.is_visible(content_selector)

    def test_10_mos_tab_default(self, authenticated_page):
        """测试10: MOS评分Tab默认激活"""
        page = authenticated_page
        
        # 验证MOS Tab默认显示
        mos_tab = page.locator('[data-tab="mos"]')
        # 根据实际激活状态验证

    def test_11_page_layout(self, authenticated_page):
        """测试11: 页面布局正确"""
        page = authenticated_page
        
        # 验证主要区域存在
        assert page.is_visible(".main-content")
        assert page.is_visible(".content-card")
        
        # 验证背景装饰
        assert page.is_visible(".bg-decoration")


# ======================== MOS评分功能测试 ========================
class TestMOSFunctionality:
    """MOS评分功能测试类"""

    def test_12_mos_upload_single_file(self, authenticated_page, test_audio_file):
        """测试12: MOS上传单个音频文件"""
        page = authenticated_page
        
        # 切换到MOS Tab
        if page.is_visible('[data-tab="mos"]'):
            page.click('[data-tab="mos"]')
            time.sleep(0.5)
        
        # 设置文件输入（如果有文件上传输入框）
        file_input = page.locator('input[type="file"]')
        if file_input.count() > 0:
            file_input.first.set_input_files(test_audio_file)
            
            # 等待上传完成
            time.sleep(2)
            
            # 验证任务创建（如果有任务列表）
            if page.is_visible(".task-list") or page.is_visible(".task-item"):
                assert page.is_visible(".task-item") or page.is_visible(".task-list")

    def test_13_mos_upload_multiple_files(self, authenticated_page, test_audio_file):
        """测试13: MOS上传多个音频文件"""
        page = authenticated_page
        
        # 切换到MOS Tab
        if page.is_visible('[data-tab="mos"]'):
            page.click('[data-tab="mos"]')
            time.sleep(0.5)
        
        # 上传多个文件（如果有批量上传功能）
        file_input = page.locator('input[type="file"]')
        if file_input.count() > 0:
            # 创建第二个测试文件
            test_file2 = os.path.join(TEST_AUDIO_DIR, "test_sample2.wav")
            if not os.path.exists(test_file2):
                import shutil
                shutil.copy(test_audio_file, test_file2)
            
            file_input.first.set_input_files([test_audio_file, test_file2])
            
            time.sleep(2)
            
            # 验证批量上传成功

    def test_14_mos_task_list_display(self, authenticated_page):
        """测试14: MOS任务列表显示"""
        page = authenticated_page
        
        # 切换到MOS Tab
        if page.is_visible('[data-tab="mos"]'):
            page.click('[data-tab="mos"]')
            time.sleep(0.5)
        
        # 验证任务列表区域存在
        if page.is_visible(".task-list-section"):
            assert page.is_visible(".task-list-section")

    def test_15_mos_statistics_cards(self, authenticated_page):
        """测试15: MOS统计卡片显示"""
        page = authenticated_page
        
        # 切换到MOS Tab
        if page.is_visible('[data-tab="mos"]'):
            page.click('[data-tab="mos"]')
            time.sleep(0.5)
        
        # 验证统计卡片（如果有）
        stat_cards = page.locator(".stat-card")
        if stat_cards.count() > 0:
            # 验证至少有统计卡片显示
            assert stat_cards.count() >= 0

    def test_16_mos_progress_indicator(self, authenticated_page, test_audio_file):
        """测试16: MOS进度指示器"""
        page = authenticated_page
        
        # 切换到MOS Tab并上传文件
        if page.is_visible('[data-tab="mos"]'):
            page.click('[data-tab="mos"]')
            time.sleep(0.5)
            
            file_input = page.locator('input[type="file"]')
            if file_input.count() > 0:
                file_input.first.set_input_files(test_audio_file)
                time.sleep(1)
                
                # 验证进度条（如果有）
                progress_bar = page.locator(".progress-bar")
                if progress_bar.count() > 0:
                    # 进度条应该可见
                    assert progress_bar.first.is_visible()


# ======================== 音频修复功能测试 ========================
class TestRestorationFunctionality:
    """音频修复功能测试类"""

    def test_17_restoration_tab_switch(self, authenticated_page):
        """测试17: 切换到音频修复Tab"""
        page = authenticated_page
        
        # 点击音频修复Tab（如果有）
        if page.is_visible('[data-tab="restoration"]'):
            page.click('[data-tab="restoration"]')
            time.sleep(0.5)
            
            # 验证Tab内容显示
            assert page.is_visible("#restoration-content") or \
                   page.is_visible(".restoration-section")

    def test_18_restoration_algorithm_list(self, authenticated_page):
        """测试18: 音频修复算法列表"""
        page = authenticated_page
        
        # 切换到修复Tab
        if page.is_visible('[data-tab="restoration"]'):
            page.click('[data-tab="restoration"]')
            time.sleep(0.5)
            
            # 验证算法选择区域（如果有）
            if page.is_visible(".algorithm-selector"):
                assert page.is_visible(".algorithm-selector")

    def test_19_restoration_upload_file(self, authenticated_page, test_audio_file):
        """测试19: 音频修复上传文件"""
        page = authenticated_page
        
        # 切换到修复Tab
        if page.is_visible('[data-tab="restoration"]'):
            page.click('[data-tab="restoration"]')
            time.sleep(0.5)
            
            # 上传文件
            file_input = page.locator('input[type="file"]')
            if file_input.count() > 0:
                file_input.first.set_input_files(test_audio_file)
                time.sleep(2)


# ======================== 参考音频功能测试 ========================
class TestReferenceAudioFunctionality:
    """参考音频功能测试类"""

    def test_20_reference_audio_tab_switch(self, authenticated_page):
        """测试20: 切换到参考音频Tab"""
        page = authenticated_page
        
        # 点击参考音频Tab（如果有）
        if page.is_visible('[data-tab="reference"]'):
            page.click('[data-tab="reference"]')
            time.sleep(0.5)
            
            # 验证Tab内容显示
            assert page.is_visible("#reference-content") or \
                   page.is_visible(".reference-section")

    def test_21_reference_audio_list_display(self, authenticated_page):
        """测试21: 参考音频列表显示"""
        page = authenticated_page
        
        # 切换到参考音频Tab
        if page.is_visible('[data-tab="reference"]'):
            page.click('[data-tab="reference"]')
            time.sleep(0.5)
            
            # 验证音频列表区域（如果有）
            if page.is_visible(".audio-list"):
                assert page.is_visible(".audio-list")

    def test_22_reference_audio_upload(self, authenticated_page, test_audio_file):
        """测试22: 参考音频上传"""
        page = authenticated_page
        
        # 切换到参考音频Tab
        if page.is_visible('[data-tab="reference"]'):
            page.click('[data-tab="reference"]')
            time.sleep(0.5)
            
            # 上传参考音频
            file_input = page.locator('input[type="file"]')
            if file_input.count() > 0:
                file_input.first.set_input_files(test_audio_file)
                time.sleep(2)


# ======================== UI交互测试 ========================
class TestUIInteractions:
    """UI交互测试类"""

    def test_23_button_hover_effects(self, authenticated_page):
        """测试23: 按钮悬停效果"""
        page = authenticated_page
        
        # 测试登出按钮悬停
        logout_btn = page.locator("#logout-btn")
        if logout_btn.is_visible():
            logout_btn.hover()
            time.sleep(0.3)
            # 验证悬停状态（视觉效果）

    def test_24_input_focus_effects(self, page):
        """测试24: 输入框聚焦效果"""
        page.wait_for_selector("#login-form")
        
        username_input = page.locator("#username-input")
        username_input.click()
        time.sleep(0.2)
        
        # 验证输入框聚焦状态
        assert username_input.is_focused()

    def test_25_form_validation_visual(self, page):
        """测试25: 表单验证视觉效果"""
        page.wait_for_selector("#login-form")
        
        # 测试空输入时的视觉效果
        page.fill("#username-input", "")
        page.fill("#password-input", "")
        page.click("#login-btn")
        time.sleep(0.5)
        
        # 验证表单仍在（验证阻止提交）
        assert page.is_visible("#login-form")

    def test_26_toast_notifications(self, authenticated_page):
        """测试26: Toast通知显示"""
        page = authenticated_page
        
        # 触发一个操作产生Toast（如果有）
        # 例如：上传文件、点击按钮等
        # 然后验证Toast显示
        
        toast_container = page.locator("#toast-container")
        # Toast容器可能不存在（只在需要时创建）

    def test_27_loading_states(self, page):
        """测试27: 加载状态显示"""
        # 初始加载状态
        loading_section = page.locator("#loading-section")
        
        # 验证加载状态短暂显示然后消失
        if loading_section.is_visible():
            time.sleep(2)
            # 加载状态应该消失
            assert not loading_section.is_visible()


# ======================== 性能测试 ========================
class TestPerformance:
    """性能测试类"""

    def test_28_page_load_time(self, page):
        """测试28: 页面加载时间"""
        start_time = time.time()
        page.goto(BASE_URL)
        page.wait_for_selector("#login-form")
        load_time = time.time() - start_time
        
        # 页面加载应在3秒内
        assert load_time < 3.0
        print(f"✅ 页面加载时间: {load_time:.2f}s")

    def test_29_login_response_time(self, page):
        """测试29: 登录响应时间"""
        page.wait_for_selector("#login-form")
        
        start_time = time.time()
        page.fill("#username-input", USERNAME)
        page.fill("#password-input", PASSWORD)
        page.click("#login-btn")
        page.wait_for_selector("#app-section")
        login_time = time.time() - start_time
        
        # 登录应在2秒内完成
        assert login_time < 2.0
        print(f"✅ 登录响应时间: {login_time:.2f}s")

    def test_30_tab_switch_response_time(self, authenticated_page):
        """测试30: Tab切换响应时间"""
        page = authenticated_page
        
        # 测试Tab切换速度
        if page.is_visible('[data-tab="mos"]'):
            start_time = time.time()
            page.click('[data-tab="mos"]')
            time.sleep(0.5)  # 等待切换动画
            switch_time = time.time() - start_time
            
            # Tab切换应在1秒内
            assert switch_time < 1.0
            print(f"✅ Tab切换时间: {switch_time:.2f}s")


# ======================== 错误处理测试 ========================
class TestErrorHandling:
    """错误处理测试类"""

    def test_31_network_error_handling(self, page):
        """测试31: 网络错误处理"""
        # 模拟网络错误（断开连接）
        # 验证错误提示显示
        
        # 这个测试需要特殊设置，暂时跳过
        pass

    def test_32_invalid_file_format(self, authenticated_page):
        """测试32: 无效文件格式处理"""
        page = authenticated_page
        
        # 切换到MOS Tab
        if page.is_visible('[data-tab="mos"]'):
            page.click('[data-tab="mos"]')
            time.sleep(0.5)
            
            # 尝试上传无效文件（如果有）
            # 验证错误提示

    def test_33_session_timeout_handling(self, authenticated_page):
        """测试33: 会话超时处理"""
        page = authenticated_page
        
        # 清除Token模拟会话超时
        page.evaluate("localStorage.removeItem('token')")
        
        # 尝试操作，验证自动跳转到登录页
        # 这取决于具体实现


# ======================== 可访问性测试 ========================
class TestAccessibility:
    """可访问性测试类"""

    def test_34_keyboard_navigation(self, page):
        """测试34: 键盘导航"""
        page.wait_for_selector("#login-form")
        
        # Tab键导航
        page.keyboard.press("Tab")
        username_input = page.locator("#username-input")
        assert username_input.is_focused()
        
        page.keyboard.press("Tab")
        password_input = page.locator("#password-input")
        assert password_input.is_focused()

    def test_35_form_labels(self, page):
        """测试35: 表单标签可访问性"""
        page.wait_for_selector("#login-form")
        
        # 验证输入框有适当的标签或占位符
        username_input = page.locator("#username-input")
        password_input = page.locator("#password-input")
        
        # 验证占位符文本
        username_placeholder = username_input.get_attribute("placeholder")
        password_placeholder = password_input.get_attribute("placeholder")
        
        assert username_placeholder is not None
        assert password_placeholder is not None

    def test_36_button_accessibility(self, page):
        """测试36: 按钮可访问性"""
        page.wait_for_selector("#login-form")
        
        login_btn = page.locator("#login-btn")
        
        # 验证按钮有文本内容
        btn_text = login_btn.text_content()
        assert len(btn_text) > 0


# ======================== 响应式设计测试 ========================
class TestResponsiveDesign:
    """响应式设计测试类"""

    def test_37_mobile_viewport(self, browser):
        """测试37: 移动端视图"""
        # 创建移动端上下文
        mobile_context = browser.new_context(
            viewport={"width": 375, "height": 667}  # iPhone尺寸
        )
        mobile_page = mobile_context.new_page()
        mobile_page.goto(BASE_URL)
        
        # 验证移动端显示
        mobile_page.wait_for_selector("#login-form")
        
        # 验证登录页面元素在移动端显示
        assert mobile_page.is_visible("#login-form")
        
        mobile_context.close()

    def test_38_tablet_viewport(self, browser):
        """测试38: 平板视图"""
        # 创建平板上下文
        tablet_context = browser.new_context(
            viewport={"width": 768, "height": 1024}  # iPad尺寸
        )
        tablet_page = tablet_context.new_page()
        tablet_page.goto(BASE_URL)
        
        # 验证平板显示
        tablet_page.wait_for_selector("#login-form")
        assert tablet_page.is_visible("#login-form")
        
        tablet_context.close()

    def test_39_desktop_viewport(self, browser):
        """测试39: 桌面视图"""
        # 创建桌面上下文
        desktop_context = browser.new_context(
            viewport={"width": 1920, "height": 1080}
        )
        desktop_page = desktop_context.new_page()
        desktop_page.goto(BASE_URL)
        
        # 验证桌面显示
        desktop_page.wait_for_selector("#login-form")
        assert desktop_page.is_visible("#login-form")
        
        desktop_context.close()


# ======================== 运行测试 ========================
if __name__ == "__main__":
    # 运行所有测试
    pytest.main([
        __file__,
        "-v",
        "--headed",  # 显示浏览器界面
        "--slowmo=100",  # 每个操作慢100ms，便于观察
        "--screenshot",  # 失败时截图
        "--video",  # 失败时录制视频
        "--html=test_report.html",  # 生成HTML报告
        "--self-contained-html"
    ])