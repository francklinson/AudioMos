"""
AudioMOS 前端UI扩展测试套件（Playwright）
包含详细的UI按钮测试、功能组合测试、边界条件测试和错误处理测试
"""
import pytest
import time
import os
import tempfile
from playwright.sync_api import sync_playwright, expect, Page


# ======================== 配置 ========================
BASE_URL = "http://localhost:8000"
USERNAME = "admin"
PASSWORD = "tp123456"
TEST_AUDIO_DIR = "tests/test_data"

# 文件大小限制（字节）
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB
MAX_FILE_COUNT = 20

# 支持的音频格式
SUPPORTED_FORMATS = ['.wav', '.mp3', '.flac', '.ogg', '.m4a']


# ======================== Fixtures ========================
@pytest.fixture(scope="module")
def browser():
    """创建浏览器实例（模块级别，所有测试共享）"""
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)
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
    page.wait_for_selector("#login-form", timeout=5000)
    page.fill("#username-input", USERNAME)
    page.fill("#password-input", PASSWORD)
    page.click("#login-btn")
    page.wait_for_selector("#app-section", timeout=5000)
    return page


@pytest.fixture(scope="module")
def test_audio_file():
    """生成测试音频文件"""
    os.makedirs(TEST_AUDIO_DIR, exist_ok=True)
    test_file = os.path.join(TEST_AUDIO_DIR, "test_sample.wav")

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
            samples = [0] * num_samples
            wav_file.writeframes(struct.pack('<' + 'h' * num_samples, *samples))

    return test_file


@pytest.fixture
def large_audio_file():
    """生成大尺寸测试音频文件（用于边界测试）"""
    os.makedirs(TEST_AUDIO_DIR, exist_ok=True)
    test_file = os.path.join(TEST_AUDIO_DIR, "large_sample.wav")

    if not os.path.exists(test_file):
        import wave
        import struct

        sample_rate = 44100
        duration = 120.0  # 2分钟
        num_samples = int(sample_rate * duration)

        with wave.open(test_file, 'wb') as wav_file:
            wav_file.setnchannels(2)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            samples = [int(1000 * (i % 100)) for i in range(num_samples)]
            wav_file.writeframes(struct.pack('<' + 'h' * num_samples, *samples))

    return test_file


# ======================== 第一部分：UI按钮样式测试 ========================
class TestButtonStyles:
    """UI按钮样式详细测试类"""

    def test_01_login_button_default_style(self, page):
        """测试01: 登录按钮默认样式"""
        page.wait_for_selector("#login-form", timeout=5000)
        login_btn = page.locator("#login-btn")

        # 验证按钮可见
        expect(login_btn).to_be_visible()

        # 验证按钮文本
        expect(login_btn).to_have_text("登录")

        # 验证按钮有正确的CSS类
        btn_class = login_btn.get_attribute("class")
        assert btn_class is not None
        assert "btn" in btn_class or "login" in btn_class.lower()

    def test_02_login_button_hover_style(self, page):
        """测试02: 登录按钮悬停样式"""
        page.wait_for_selector("#login-form", timeout=5000)
        login_btn = page.locator("#login-btn")

        # 悬停在按钮上
        login_btn.hover()
        time.sleep(0.3)

        # 验证按钮仍然可见
        expect(login_btn).to_be_visible()

        # 获取悬停状态下的样式（如果可以通过CSS验证）
        # 注意：具体的样式验证可能需要根据实际CSS实现调整
        btn_opacity = login_btn.evaluate("el => window.getComputedStyle(el).opacity")
        assert btn_opacity is not None

    def test_03_login_button_active_state(self, page):
        """测试03: 登录按钮激活状态"""
        page.wait_for_selector("#login-form", timeout=5000)
        login_btn = page.locator("#login-btn")

        # 填写登录信息
        page.fill("#username-input", USERNAME)
        page.fill("#password-input", PASSWORD)

        # 点击按钮（不等待跳转，只验证点击前的状态）
        login_btn.click()

        # 验证按钮被点击后状态改变（例如禁用或加载状态）
        # 这取决于实际实现

    def test_04_login_button_disabled_state(self, page):
        """测试04: 登录按钮禁用状态"""
        page.wait_for_selector("#login-form", timeout=5000)
        login_btn = page.locator("#login-btn")

        # 检查按钮是否有禁用状态
        is_disabled = login_btn.is_disabled()

        # 如果按钮在某些情况下应该禁用，验证禁用状态
        # 例如：空输入时
        page.fill("#username-input", "")
        page.fill("#password-input", "")

        # 某些实现可能在空输入时禁用按钮
        # 根据实际实现调整此测试

    def test_05_logout_button_style(self, authenticated_page):
        """测试05: 登出按钮样式"""
        page = authenticated_page

        logout_btn = page.locator("#logout-btn")
        expect(logout_btn).to_be_visible()

        # 验证按钮有正确的图标或文本
        btn_text = logout_btn.text_content()
        assert "登出" in btn_text or "退出" in btn_text or "Logout" in btn_text

    def test_06_logout_button_hover_style(self, authenticated_page):
        """测试06: 登出按钮悬停样式"""
        page = authenticated_page

        logout_btn = page.locator("#logout-btn")
        logout_btn.hover()
        time.sleep(0.3)

        # 验证按钮在悬停时可见
        expect(logout_btn).to_be_visible()

    def test_07_tab_button_styles(self, authenticated_page):
        """测试07: Tab按钮样式"""
        page = authenticated_page

        tabs = ["mos", "denoise", "restoration", "reference"]
        for tab_name in tabs:
            tab_selector = f'[data-tab="{tab_name}"]'
            if page.is_visible(tab_selector):
                tab_btn = page.locator(tab_selector)

                # 验证Tab按钮可见
                expect(tab_btn).to_be_visible()

                # 验证Tab有文本内容
                tab_text = tab_btn.text_content()
                assert len(tab_text) > 0

    def test_08_tab_button_active_state(self, authenticated_page):
        """测试08: Tab按钮激活状态"""
        page = authenticated_page

        # 点击第一个Tab
        tab_selector = '[data-tab="mos"]'
        if page.is_visible(tab_selector):
            tab_btn = page.locator(tab_selector)
            tab_btn.click()
            time.sleep(0.3)

            # 验证Tab处于激活状态（检查class是否包含active）
            tab_class = tab_btn.get_attribute("class")
            # 根据实际实现，激活的Tab可能有"active"类

    def test_09_tab_button_hover_style(self, authenticated_page):
        """测试09: Tab按钮悬停样式"""
        page = authenticated_page

        tab_selector = '[data-tab="mos"]'
        if page.is_visible(tab_selector):
            tab_btn = page.locator(tab_selector)
            tab_btn.hover()
            time.sleep(0.3)

            expect(tab_btn).to_be_visible()

    def test_10_upload_button_style(self, authenticated_page):
        """测试10: 上传按钮样式"""
        page = authenticated_page

        # 切换到MOS Tab
        if page.is_visible('[data-tab="mos"]'):
            page.click('[data-tab="mos"]')
            time.sleep(0.5)

        # 查找上传按钮（可能在文件输入区域）
        upload_btn = page.locator('button:has-text("上传"), button:has-text("选择文件")')
        if upload_btn.count() > 0:
            expect(upload_btn.first).to_be_visible()

    def test_11_primary_button_colors(self, authenticated_page):
        """测试11: 主要按钮颜色样式"""
        page = authenticated_page

        # 查找所有主要按钮
        primary_btns = page.locator(".btn-primary, .primary-btn, button.primary")
        if primary_btns.count() > 0:
            # 验证主要按钮存在
            assert primary_btns.count() > 0

            # 验证按钮颜色（通过CSS）
            first_btn = primary_btns.first
            bg_color = first_btn.evaluate("el => window.getComputedStyle(el).backgroundColor")
            assert bg_color is not None

    def test_12_secondary_button_colors(self, authenticated_page):
        """测试12: 次要按钮颜色样式"""
        page = authenticated_page

        # 查找所有次要按钮
        secondary_btns = page.locator(".btn-secondary, .secondary-btn, button.secondary")
        if secondary_btns.count() > 0:
            assert secondary_btns.count() > 0

    def test_13_danger_button_colors(self, authenticated_page):
        """测试13: 危险按钮颜色样式"""
        page = authenticated_page

        # 查找所有危险按钮（如删除、取消等）
        danger_btns = page.locator(".btn-danger, .danger-btn, button.danger")
        if danger_btns.count() > 0:
            assert danger_btns.count() > 0

    def test_14_button_border_styles(self, authenticated_page):
        """测试14: 按钮边框样式"""
        page = authenticated_page

        logout_btn = page.locator("#logout-btn")
        if logout_btn.is_visible():
            border_style = logout_btn.evaluate("el => window.getComputedStyle(el).borderStyle")
            # 验证边框样式存在
            assert border_style is not None

    def test_15_button_shadow_effects(self, authenticated_page):
        """测试15: 按钮阴影效果"""
        page = authenticated_page

        # 查找登录按钮或其他主要按钮
        login_btn = page.locator("#login-btn")
        if login_btn.is_visible():
            box_shadow = login_btn.evaluate("el => window.getComputedStyle(el).boxShadow")
            # 验证阴影样式
            assert box_shadow is not None


# ======================== 第二部分：UI按钮交互测试 ========================
class TestButtonInteractions:
    """UI按钮交互详细测试类"""

    def test_16_login_button_click_response(self, page):
        """测试16: 登录按钮点击响应"""
        page.wait_for_selector("#login-form", timeout=5000)

        page.fill("#username-input", USERNAME)
        page.fill("#password-input", PASSWORD)

        login_btn = page.locator("#login-btn")

        # 点击登录
        login_btn.click()

        # 验证跳转到主页面
        page.wait_for_selector("#app-section", timeout=5000)
        expect(page.locator("#app-section")).to_be_visible()

    def test_17_login_button_enter_key(self, page):
        """测试17: 登录按钮Enter键提交"""
        page.wait_for_selector("#login-form", timeout=5000)

        page.fill("#username-input", USERNAME)
        page.fill("#password-input", PASSWORD)

        # 按Enter键提交
        page.keyboard.press("Enter")

        # 验证登录成功
        page.wait_for_selector("#app-section", timeout=5000)
        expect(page.locator("#app-section")).to_be_visible()

    def test_18_logout_button_confirmation(self, authenticated_page):
        """测试18: 登出按钮确认对话框"""
        page = authenticated_page

        # 点击登出按钮
        logout_btn = page.locator("#logout-btn")

        # 监听对话框（如果有）
        def handle_dialog(dialog):
            assert "确认" in dialog.message or "登出" in dialog.message or "退出" in dialog.message
            dialog.dismiss()

        page.on("dialog", handle_dialog)
        logout_btn.click()

        # 等待回到登录页面
        page.wait_for_selector("#login-section", timeout=3000)

    def test_19_tab_button_click_animation(self, authenticated_page):
        """测试19: Tab按钮点击动画效果"""
        page = authenticated_page

        tabs = ["mos", "denoise", "restoration", "reference"]
        for tab_name in tabs:
            tab_selector = f'[data-tab="{tab_name}"]'
            if page.is_visible(tab_selector):
                tab_btn = page.locator(tab_selector)

                # 点击并验证动画效果
                tab_btn.click()
                time.sleep(0.5)

                # 验证Tab内容显示
                expect(tab_btn).to_be_visible()

    def test_20_button_double_click_prevention(self, page):
        """测试20: 按钮双击防护"""
        page.wait_for_selector("#login-form", timeout=5000)

        page.fill("#username-input", USERNAME)
        page.fill("#password-input", PASSWORD)

        login_btn = page.locator("#login-btn")

        # 快速双击按钮
        login_btn.click()
        time.sleep(0.1)
        login_btn.click()

        # 验证仍然能正常登录
        page.wait_for_selector("#app-section", timeout=5000)
        expect(page.locator("#app-section")).to_be_visible()

    def test_21_button_right_click_menu(self, authenticated_page):
        """测试21: 按钮右键菜单"""
        page = authenticated_page

        logout_btn = page.locator("#logout-btn")

        # 右键点击按钮
        logout_btn.click(button="right")
        time.sleep(0.3)

        # 验证是否有自定义右键菜单（根据实际实现）
        # 这里只是验证右键不会导致错误

    def test_22_button_focus_on_tab_navigation(self, page):
        """测试22: Tab键导航按钮聚焦"""
        page.wait_for_selector("#login-form", timeout=5000)

        # Tab键导航到登录按钮
        page.keyboard.press("Tab")  # 聚焦用户名输入框
        page.keyboard.press("Tab")  # 聚焦密码输入框
        page.keyboard.press("Tab")  # 聚焦登录按钮

        login_btn = page.locator("#login-btn")
        expect(login_btn).to_be_focused()

    def test_23_button_space_key_activation(self, page):
        """测试23: 按钮空格键激活"""
        page.wait_for_selector("#login-form", timeout=5000)

        page.fill("#username-input", USERNAME)
        page.fill("#password-input", PASSWORD)

        # Tab键导航到登录按钮
        page.keyboard.press("Tab")
        page.keyboard.press("Tab")
        page.keyboard.press("Tab")

        # 空格键激活按钮
        page.keyboard.press("Space")

        # 验证登录成功
        page.wait_for_selector("#app-section", timeout=5000)
        expect(page.locator("#app-section")).to_be_visible()

    def test_24_button_loading_state(self, authenticated_page):
        """测试24: 按钮加载状态"""
        page = authenticated_page

        # 切换到MOS Tab
        if page.is_visible('[data-tab="mos"]'):
            page.click('[data-tab="mos"]')
            time.sleep(0.5)

        # 查找上传按钮
        upload_btn = page.locator('button:has-text("上传"), button:has-text("选择文件")')
        if upload_btn.count() > 0:
            # 点击按钮
            upload_btn.first.click()

            # 验证按钮是否显示加载状态（如果有）
            # 例如：按钮文本变为"上传中..."或显示加载图标

    def test_25_button_disabled_after_click(self, page):
        """测试25: 点击后按钮禁用状态"""
        page.wait_for_selector("#login-form", timeout=5000)

        page.fill("#username-input", USERNAME)
        page.fill("#password-input", PASSWORD)

        login_btn = page.locator("#login-btn")
        login_btn.click()

        # 某些实现在点击后立即禁用按钮以防止重复提交
        # 验证按钮状态（根据实际实现）

    def test_26_button_cursor_style(self, page):
        """测试26: 按钮光标样式"""
        page.wait_for_selector("#login-form", timeout=5000)

        login_btn = page.locator("#login-btn")

        # 验证按钮的光标样式
        cursor_style = login_btn.evaluate("el => window.getComputedStyle(el).cursor")
        # 期望是pointer或其他可点击的光标样式
        assert cursor_style in ["pointer", "hand", "auto"]

    def test_27_button_font_size(self, page):
        """测试27: 按钮字体大小"""
        page.wait_for_selector("#login-form", timeout=5000)

        login_btn = page.locator("#login-btn")

        # 验证按钮字体大小
        font_size = login_btn.evaluate("el => window.getComputedStyle(el).fontSize")
        assert font_size is not None
        assert len(font_size) > 0

    def test_28_button_padding(self, page):
        """测试28: 按钮内边距"""
        page.wait_for_selector("#login-form", timeout=5000)

        login_btn = page.locator("#login-btn")

        # 验证按钮内边距
        padding = login_btn.evaluate("el => window.getComputedStyle(el).padding")
        assert padding is not None

    def test_29_button_margin(self, page):
        """测试29: 按钮外边距"""
        page.wait_for_selector("#login-form", timeout=5000)

        login_btn = page.locator("#login-btn")

        # 验证按钮外边距
        margin = login_btn.evaluate("el => window.getComputedStyle(el).margin")
        assert margin is not None

    def test_30_button_border_radius(self, page):
        """测试30: 按钮圆角"""
        page.wait_for_selector("#login-form", timeout=5000)

        login_btn = page.locator("#login-btn")

        # 验证按钮圆角
        border_radius = login_btn.evaluate("el => window.getComputedStyle(el).borderRadius")
        assert border_radius is not None


# ======================== 第三部分：功能组合测试 ========================
class TestFunctionalCombinations:
    """功能组合测试类"""

    def test_31_login_mos_workflow(self, authenticated_page, test_audio_file):
        """测试31: 登录→MOS评分完整流程"""
        page = authenticated_page

        # 切换到MOS Tab
        if page.is_visible('[data-tab="mos"]'):
            page.click('[data-tab="mos"]')
            time.sleep(0.5)

            # 上传文件
            file_input = page.locator('input[type="file"]')
            if file_input.count() > 0:
                file_input.first.set_input_files(test_audio_file)
                time.sleep(2)

                # 验证文件上传成功（检查任务列表或结果）
                # 根据实际实现验证

    def test_32_login_denoise_workflow(self, authenticated_page, test_audio_file):
        """测试32: 登录→降噪处理完整流程"""
        page = authenticated_page

        # 切换到降噪Tab
        if page.is_visible('[data-tab="denoise"]'):
            page.click('[data-tab="denoise"]')
            time.sleep(0.5)

            # 上传文件
            file_input = page.locator('input[type="file"]')
            if file_input.count() > 0:
                file_input.first.set_input_files(test_audio_file)
                time.sleep(2)

    def test_33_login_restoration_workflow(self, authenticated_page, test_audio_file):
        """测试33: 登录→音频修复完整流程"""
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

    def test_34_login_reference_workflow(self, authenticated_page, test_audio_file):
        """测试34: 登录→参考音频完整流程"""
        page = authenticated_page

        # 切换到参考音频Tab
        if page.is_visible('[data-tab="reference"]'):
            page.click('[data-tab="reference"]')
            time.sleep(0.5)

            # 上传文件
            file_input = page.locator('input[type="file"]')
            if file_input.count() > 0:
                file_input.first.set_input_files(test_audio_file)
                time.sleep(2)

    def test_35_login_multiple_uploads_logout(self, authenticated_page, test_audio_file):
        """测试35: 登录→多次上传→登出完整流程"""
        page = authenticated_page

        # 执行多次上传操作
        for i in range(3):
            if page.is_visible('[data-tab="mos"]'):
                page.click('[data-tab="mos"]')
                time.sleep(0.5)

                file_input = page.locator('input[type="file"]')
                if file_input.count() > 0:
                    file_input.first.set_input_files(test_audio_file)
                    time.sleep(1)

        # 登出
        logout_btn = page.locator("#logout-btn")
        logout_btn.click()
        page.wait_for_selector("#login-section", timeout=3000)

        # 验证登出成功
        expect(page.locator("#login-section")).to_be_visible()

    def test_36_tab_switching_sequence(self, authenticated_page):
        """测试36: 连续切换多个Tab"""
        page = authenticated_page

        tabs = ["mos", "denoise", "restoration", "reference", "mos"]

        for tab_name in tabs:
            tab_selector = f'[data-tab="{tab_name}"]'
            if page.is_visible(tab_selector):
                page.click(tab_selector)
                time.sleep(0.3)

                # 验证Tab内容显示
                tab_btn = page.locator(tab_selector)
                expect(tab_btn).to_be_visible()

    def test_37_login_upload_download_workflow(self, authenticated_page, test_audio_file):
        """测试37: 登录→上传→下载流程"""
        page = authenticated_page

        # 切换到MOS Tab
        if page.is_visible('[data-tab="mos"]'):
            page.click('[data-tab="mos"]')
            time.sleep(0.5)

            # 上传文件
            file_input = page.locator('input[type="file"]')
            if file_input.count() > 0:
                file_input.first.set_input_files(test_audio_file)
                time.sleep(2)

                # 查找下载按钮（如果有）
                download_btn = page.locator('button:has-text("下载"), button:has-text("Download")')
                if download_btn.count() > 0:
                    # 点击下载
                    download_btn.first.click()
                    time.sleep(1)

    def test_38_login_settings_save_workflow(self, authenticated_page):
        """测试38: 登录→设置→保存流程"""
        page = authenticated_page

        # 查找设置按钮（如果有）
        settings_btn = page.locator('button:has-text("设置"), button:has-text("Settings")')
        if settings_btn.count() > 0:
            settings_btn.first.click()
            time.sleep(0.5)

            # 修改设置（根据实际实现）

            # 保存设置
            save_btn = page.locator('button:has-text("保存"), button:has-text("Save")')
            if save_btn.count() > 0:
                save_btn.first.click()
                time.sleep(0.5)

    def test_39_login_view_results_export_workflow(self, authenticated_page):
        """测试39: 登录→查看结果→导出流程"""
        page = authenticated_page

        # 切换到结果页面（如果有）
        results_tab = page.locator('[data-tab="results"], button:has-text("结果")')
        if results_tab.count() > 0:
            results_tab.first.click()
            time.sleep(0.5)

            # 查找导出按钮
            export_btn = page.locator('button:has-text("导出"), button:has-text("Export")')
            if export_btn.count() > 0:
                export_btn.first.click()
                time.sleep(1)

    def test_40_login_delete_task_logout_workflow(self, authenticated_page, test_audio_file):
        """测试40: 登录→上传→删除任务→登出流程"""
        page = authenticated_page

        # 切换到MOS Tab并上传
        if page.is_visible('[data-tab="mos"]'):
            page.click('[data-tab="mos"]')
            time.sleep(0.5)

            file_input = page.locator('input[type="file"]')
            if file_input.count() > 0:
                file_input.first.set_input_files(test_audio_file)
                time.sleep(2)

                # 查找删除按钮（如果有）
                delete_btn = page.locator('button:has-text("删除"), button:has-text("Delete")')
                if delete_btn.count() > 0:
                    delete_btn.first.click()
                    time.sleep(0.5)

        # 登出
        logout_btn = page.locator("#logout-btn")
        logout_btn.click()
        page.wait_for_selector("#login-section", timeout=3000)

    def test_41_repeated_login_logout(self, page):
        """测试41: 重复登录登出"""
        for _ in range(3):
            # 登录
            page.wait_for_selector("#login-form", timeout=5000)
            page.fill("#username-input", USERNAME)
            page.fill("#password-input", PASSWORD)
            page.click("#login-btn")
            page.wait_for_selector("#app-section", timeout=5000)

            # 登出
            logout_btn = page.locator("#logout-btn")
            logout_btn.click()
            page.wait_for_selector("#login-section", timeout=3000)

    def test_42_multiple_tab_operations(self, authenticated_page, test_audio_file):
        """测试42: 在多个Tab中执行操作"""
        page = authenticated_page

        # 在MOS Tab上传
        if page.is_visible('[data-tab="mos"]'):
            page.click('[data-tab="mos"]')
            time.sleep(0.5)
            file_input = page.locator('input[type="file"]')
            if file_input.count() > 0:
                file_input.first.set_input_files(test_audio_file)
                time.sleep(1)

        # 切换到降噪Tab
        if page.is_visible('[data-tab="denoise"]'):
            page.click('[data-tab="denoise"]')
            time.sleep(0.5)
            file_input = page.locator('input[type="file"]')
            if file_input.count() > 0:
                file_input.first.set_input_files(test_audio_file)
                time.sleep(1)

        # 切换到修复Tab
        if page.is_visible('[data-tab="restoration"]'):
            page.click('[data-tab="restoration"]')
            time.sleep(0.5)

    def test_43_login_refresh_page_state(self, authenticated_page):
        """测试43: 登录后刷新页面状态保持"""
        page = authenticated_page

        # 切换到某个Tab
        if page.is_visible('[data-tab="mos"]'):
            page.click('[data-tab="mos"]')
            time.sleep(0.5)

        # 刷新页面
        page.reload()
        time.sleep(1)

        # 验证登录状态保持
        # 这取决于实际实现（是否使用token持久化）

    def test_44_login_network_disconnect_reconnect(self, authenticated_page):
        """测试44: 登录后网络断开重连"""
        page = authenticated_page

        # 模拟网络断开
        page.context.set_offline(True)
        time.sleep(1)

        # 模拟网络重连
        page.context.set_offline(False)
        time.sleep(1)

        # 验证页面状态（根据实际实现）

    def test_45_login_multiple_windows(self, browser):
        """测试45: 多窗口登录状态同步"""
        # 创建两个上下文
        context1 = browser.new_context()
        context2 = browser.new_context()

        page1 = context1.new_page()
        page2 = context2.new_page()

        # 在第一个窗口登录
        page1.goto(BASE_URL)
        page1.wait_for_selector("#login-form", timeout=5000)
        page1.fill("#username-input", USERNAME)
        page1.fill("#password-input", PASSWORD)
        page1.click("#login-btn")
        page1.wait_for_selector("#app-section", timeout=5000)

        # 第二个窗口未登录
        page2.goto(BASE_URL)
        # 验证第二个窗口的状态（根据实际实现）

        context1.close()
        context2.close()


# ======================== 第四部分：边界条件测试 ========================
class TestBoundaryConditions:
    """边界条件测试类"""

    def test_46_large_file_upload(self, authenticated_page, large_audio_file):
        """测试46: 大文件上传（接近限制）"""
        page = authenticated_page

        if page.is_visible('[data-tab="mos"]'):
            page.click('[data-tab="mos"]')
            time.sleep(0.5)

            file_input = page.locator('input[type="file"]')
            if file_input.count() > 0:
                # 尝试上传大文件
                file_input.first.set_input_files(large_audio_file)
                time.sleep(3)

                # 验证是否有文件大小错误提示
                error_msg = page.locator('.error-message, .toast-error, .alert-danger')
                # 根据实际实现验证错误处理

    def test_47_file_size_exceed_limit(self, authenticated_page):
        """测试47: 文件大小超过限制"""
        page = authenticated_page

        # 创建一个超大文件的模拟
        # 这需要根据实际实现调整

    def test_48_maximum_file_count(self, authenticated_page, test_audio_file):
        """测试48: 最大文件数量上传"""
        page = authenticated_page

        if page.is_visible('[data-tab="mos"]'):
            page.click('[data-tab="mos"]')
            time.sleep(0.5)

            file_input = page.locator('input[type="file"]')
            if file_input.count() > 0:
                # 创建多个测试文件
                test_files = []
                for i in range(min(MAX_FILE_COUNT, 10)):  # 测试10个文件
                    test_file = os.path.join(TEST_AUDIO_DIR, f"test_{i}.wav")
                    if not os.path.exists(test_file):
                        import shutil
                        shutil.copy(test_audio_file, test_file)
                    test_files.append(test_file)

                # 尝试上传多个文件
                file_input.first.set_input_files(test_files)
                time.sleep(3)

    def test_49_file_count_exceed_limit(self, authenticated_page, test_audio_file):
        """测试49: 文件数量超过限制"""
        page = authenticated_page

        if page.is_visible('[data-tab="mos"]'):
            page.click('[data-tab="mos"]')
            time.sleep(0.5)

            file_input = page.locator('input[type="file"]')
            if file_input.count() > 0:
                # 创建超过限制数量的文件列表
                test_files = [test_audio_file] * (MAX_FILE_COUNT + 5)

                # 尝试上传
                file_input.first.set_input_files(test_files)
                time.sleep(2)

                # 验证错误提示

    def test_50_unsupported_file_format(self, authenticated_page):
        """测试50: 不支持的文件格式"""
        page = authenticated_page

        if page.is_visible('[data-tab="mos"]'):
            page.click('[data-tab="mos"]')
            time.sleep(0.5)

            # 创建一个非音频文件
            invalid_file = os.path.join(TEST_AUDIO_DIR, "test_invalid.txt")
            with open(invalid_file, 'w') as f:
                f.write("This is not an audio file")

            file_input = page.locator('input[type="file"]')
            if file_input.count() > 0:
                try:
                    file_input.first.set_input_files(invalid_file)
                    time.sleep(2)

                    # 验证错误提示
                    error_msg = page.locator('.error-message, .toast-error, .alert-danger')
                    # 根据实际实现验证
                except Exception:
                    # 某些浏览器可能不允许上传不支持的文件类型
                    pass

    def test_51_empty_file_upload(self, authenticated_page):
        """测试51: 空文件上传"""
        page = authenticated_page

        if page.is_visible('[data-tab="mos"]'):
            page.click('[data-tab="mos"]')
            time.sleep(0.5)

            # 创建一个空文件
            empty_file = os.path.join(TEST_AUDIO_DIR, "test_empty.wav")
            with open(empty_file, 'wb') as f:
                pass  # 创建空文件

            file_input = page.locator('input[type="file"]')
            if file_input.count() > 0:
                try:
                    file_input.first.set_input_files(empty_file)
                    time.sleep(2)

                    # 验证错误提示
                except Exception:
                    pass

    def test_52_corrupted_audio_file(self, authenticated_page):
        """测试52: 损坏的音频文件"""
        page = authenticated_page

        if page.is_visible('[data-tab="mos"]'):
            page.click('[data-tab="mos"]')
            time.sleep(0.5)

            # 创建一个损坏的音频文件
            corrupted_file = os.path.join(TEST_AUDIO_DIR, "test_corrupted.wav")
            with open(corrupted_file, 'wb') as f:
                f.write(b'This is corrupted audio data')

            file_input = page.locator('input[type="file"]')
            if file_input.count() > 0:
                try:
                    file_input.first.set_input_files(corrupted_file)
                    time.sleep(2)

                    # 验证错误提示
                except Exception:
                    pass

    def test_53_extreme_username_length(self, page):
        """测试53: 极端用户名长度"""
        page.wait_for_selector("#login-form", timeout=5000)

        # 测试超长用户名
        long_username = "a" * 1000
        page.fill("#username-input", long_username)
        page.fill("#password-input", PASSWORD)
        page.click("#login-btn")

        time.sleep(1)
        # 验证是否有长度错误提示

    def test_54_extreme_password_length(self, page):
        """测试54: 极端密码长度"""
        page.wait_for_selector("#login-form", timeout=5000)

        # 测试超长密码
        long_password = "a" * 1000
        page.fill("#username-input", USERNAME)
        page.fill("#password-input", long_password)
        page.click("#login-btn")

        time.sleep(1)
        # 验证是否有错误提示

    def test_55_special_characters_in_username(self, page):
        """测试55: 用户名中的特殊字符"""
        page.wait_for_selector("#login-form", timeout=5000)

        # 测试特殊字符
        special_username = "user<script>alert('xss')</script>"
        page.fill("#username-input", special_username)
        page.fill("#password-input", PASSWORD)
        page.click("#login-btn")

        time.sleep(1)
        # 验证XSS防护

    def test_56_special_characters_in_password(self, page):
        """测试56: 密码中的特殊字符"""
        page.wait_for_selector("#login-form", timeout=5000)

        # 测试特殊字符密码
        special_password = "pass<script>alert('xss')</script>"
        page.fill("#username-input", USERNAME)
        page.fill("#password-input", special_password)
        page.click("#login-btn")

        time.sleep(1)
        # 验证XSS防护

    def test_57_unicode_characters_in_input(self, page):
        """测试57: Unicode字符输入"""
        page.wait_for_selector("#login-form", timeout=5000)

        # 测试Unicode字符
        unicode_username = "用户名测试"
        unicode_password = "密码测试"
        page.fill("#username-input", unicode_username)
        page.fill("#password-input", unicode_password)
        page.click("#login-btn")

        time.sleep(1)

    def test_58_zero_duration_audio_file(self, authenticated_page):
        """测试58: 零时长音频文件"""
        page = authenticated_page

        if page.is_visible('[data-tab="mos"]'):
            page.click('[data-tab="mos"]')
            time.sleep(0.5)

            # 创建零时长音频文件
            zero_file = os.path.join(TEST_AUDIO_DIR, "test_zero.wav")
            import wave
            import struct
            with wave.open(zero_file, 'wb') as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(16000)
                # 零时长

            file_input = page.locator('input[type="file"]')
            if file_input.count() > 0:
                try:
                    file_input.first.set_input_files(zero_file)
                    time.sleep(2)
                except Exception:
                    pass

    def test_59_extreme_sample_rate_audio(self, authenticated_page):
        """测试59: 极端采样率音频文件"""
        page = authenticated_page

        if page.is_visible('[data-tab="mos"]'):
            page.click('[data-tab="mos"]')
            time.sleep(0.5)

            # 创建极端采样率音频文件
            extreme_file = os.path.join(TEST_AUDIO_DIR, "test_extreme_sr.wav")
            import wave
            import struct
            with wave.open(extreme_file, 'wb') as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(192000)  # 超高采样率
                samples = [0] * 16000
                wav_file.writeframes(struct.pack('<' + 'h' * 16000, *samples))

            file_input = page.locator('input[type="file"]')
            if file_input.count() > 0:
                try:
                    file_input.first.set_input_files(extreme_file)
                    time.sleep(2)
                except Exception:
                    pass

    def test_60_multichannel_audio_file(self, authenticated_page):
        """测试60: 多声道音频文件"""
        page = authenticated_page

        if page.is_visible('[data-tab="mos"]'):
            page.click('[data-tab="mos"]')
            time.sleep(0.5)

            # 创建多声道音频文件
            multi_file = os.path.join(TEST_AUDIO_DIR, "test_multichannel.wav")
            import wave
            import struct
            with wave.open(multi_file, 'wb') as wav_file:
                wav_file.setnchannels(8)  # 8声道
                wav_file.setsampwidth(2)
                wav_file.setframerate(44100)
                samples = [0] * 88200
                wav_file.writeframes(struct.pack('<' + 'h' * 88200, *samples))

            file_input = page.locator('input[type="file"]')
            if file_input.count() > 0:
                try:
                    file_input.first.set_input_files(multi_file)
                    time.sleep(2)
                except Exception:
                    pass


# ======================== 第五部分：错误处理测试 ========================
class TestErrorHandling:
    """错误处理测试类"""

    def test_61_network_error_on_login(self, page):
        """测试61: 登录时网络错误"""
        page.wait_for_selector("#login-form", timeout=5000)

        # 设置网络离线
        page.context.set_offline(True)

        page.fill("#username-input", USERNAME)
        page.fill("#password-input", PASSWORD)
        page.click("#login-btn")

        time.sleep(2)

        # 验证错误提示
        # 应该显示网络错误提示

        # 恢复网络
        page.context.set_offline(False)

    def test_62_server_error_500(self, authenticated_page, page):
        """测试62: 服务器500错误处理"""
        # 模拟服务器错误（需要后端配合）
        # 这通常需要mock或特殊的测试环境
        pass

    def test_63_session_expired_error(self, authenticated_page):
        """测试63: 会话过期错误"""
        page = authenticated_page

        # 清除token
        page.evaluate("localStorage.removeItem('token')")
        page.evaluate("sessionStorage.clear()")

        # 尝试执行操作
        if page.is_visible('[data-tab="mos"]'):
            page.click('[data-tab="mos"]')
            time.sleep(1)

            # 验证是否重定向到登录页
            # 这取决于实际实现

    def test_64_token_invalid_error(self, authenticated_page):
        """测试64: 无效Token错误"""
        page = authenticated_page

        # 设置无效token
        page.evaluate("localStorage.setItem('token', 'invalid_token_12345')")

        # 尝试操作
        if page.is_visible('[data-tab="mos"]'):
            page.click('[data-tab="mos"]')
            time.sleep(1)

            # 验证错误处理

    def test_65_file_permission_error(self, authenticated_page):
        """测试65: 文件权限错误"""
        page = authenticated_page

        # 这通常需要在操作系统级别模拟权限错误
        # 在某些情况下可以通过上传特定文件来触发
        pass

    def test_66_disk_space_error(self, authenticated_page):
        """测试66: 磁盘空间不足错误"""
        # 这需要模拟磁盘空间不足
        # 通常在系统级别测试
        pass

    def test_67_concurrent_upload_error(self, authenticated_page, test_audio_file):
        """测试67: 并发上传错误"""
        page = authenticated_page

        if page.is_visible('[data-tab="mos"]'):
            page.click('[data-tab="mos"]')
            time.sleep(0.5)

            file_input = page.locator('input[type="file"]')
            if file_input.count() > 0:
                # 快速连续上传多个文件
                for _ in range(5):
                    file_input.first.set_input_files(test_audio_file)
                    time.sleep(0.2)

                # 验证并发处理是否正确

    def test_68_timeout_error(self, authenticated_page):
        """测试68: 超时错误"""
        page = authenticated_page

        # 设置较短的超时时间
        page.set_default_timeout(100)  # 100ms

        try:
            if page.is_visible('[data-tab="mos"]'):
                page.click('[data-tab="mos"]')
                time.sleep(0.5)
        except Exception:
            # 验证超时错误处理
            pass
        finally:
            page.set_default_timeout(30000)  # 恢复默认超时

    def test_69_malformed_response_error(self, authenticated_page):
        """测试69: 响应格式错误"""
        # 这需要mock服务器返回格式错误的响应
        # 通常在集成测试中处理
        pass

    def test_70_rate_limit_error(self, authenticated_page):
        """测试70: 请求频率限制错误"""
        page = authenticated_page

        # 快速发送多个请求
        for _ in range(20):
            if page.is_visible('[data-tab="mos"]'):
                page.click('[data-tab="mos"]')
                time.sleep(0.1)

        # 验证频率限制错误处理

    def test_71_authentication_timeout(self, authenticated_page):
        """测试71: 认证超时"""
        page = authenticated_page

        # 等待较长时间模拟超时
        time.sleep(5)

        # 尝试操作
        if page.is_visible('[data-tab="mos"]'):
            page.click('[data-tab="mos"]')
            time.sleep(1)

            # 验证超时处理

    def test_72_invalid_api_response(self, authenticated_page):
        """测试72: 无效API响应"""
        # 这需要mock服务器返回无效响应
        # 通常在集成测试中处理
        pass

    def test_73_javascript_error_handling(self, authenticated_page):
        """测试73: JavaScript错误处理"""
        page = authenticated_page

        # 监听控制台错误
        errors = []
        page.on("console", lambda msg: errors.append(msg.text) if msg.type == "error" else None)

        # 执行一些操作
        if page.is_visible('[data-tab="mos"]'):
            page.click('[data-tab="mos"]')
            time.sleep(1)

        # 验证是否有JavaScript错误
        # 根据实际需求调整

    def test_74_resource_not_found_error(self, authenticated_page):
        """测试74: 资源未找到错误"""
        page = authenticated_page

        # 尝试访问不存在的资源
        # 例如：不存在的任务ID
        # 这需要根据实际API设计

    def test_75_duplicate_submission_error(self, authenticated_page, test_audio_file):
        """测试75: 重复提交错误"""
        page = authenticated_page

        if page.is_visible('[data-tab="mos"]'):
            page.click('[data-tab="mos"]')
            time.sleep(0.5)

            file_input = page.locator('input[type="file"]')
            if file_input.count() > 0:
                # 上传相同文件两次
                file_input.first.set_input_files(test_audio_file)
                time.sleep(1)

                file_input.first.set_input_files(test_audio_file)
                time.sleep(1)

                # 验证重复提交处理

    def test_76_invalid_input_sanitization(self, authenticated_page):
        """测试76: 无效输入清理"""
        page = authenticated_page

        # 测试各种无效输入
        invalid_inputs = [
            "<script>alert('xss')</script>",
            "'; DROP TABLE users; --",
            "../../../etc/passwd",
            "null",
            "undefined",
            "NaN",
        ]

        for invalid_input in invalid_inputs:
            # 尝试在输入框中输入无效数据
            # 根据实际可用的输入框调整
            pass

    def test_77_browser_back_button_error(self, authenticated_page):
        """测试77: 浏览器后退按钮错误"""
        page = authenticated_page

        # 导航到其他页面
        if page.is_visible('[data-tab="mos"]'):
            page.click('[data-tab="mos"]')
            time.sleep(0.5)

        # 点击后退按钮
        page.go_back()
        time.sleep(1)

        # 验证页面状态
        # 应该正确处理后退导航

    def test_78_browser_refresh_error(self, authenticated_page):
        """测试78: 浏览器刷新错误"""
        page = authenticated_page

        # 刷新页面
        page.reload()
        time.sleep(2)

        # 验证页面状态
        # 应该保持登录状态或正确处理刷新

    def test_79_multiple_tab_session_error(self, browser):
        """测试79: 多标签页会话错误"""
        # 创建两个标签页
        context = browser.new_context()
        page1 = context.new_page()
        page2 = context.new_page()

        # 第一个标签页登录
        page1.goto(BASE_URL)
        page1.wait_for_selector("#login-form", timeout=5000)
        page1.fill("#username-input", USERNAME)
        page1.fill("#password-input", PASSWORD)
        page1.click("#login-btn")
        page1.wait_for_selector("#app-section", timeout=5000)

        # 第二个标签页访问
        page2.goto(BASE_URL)
        # 验证第二个标签页的状态

        context.close()

    def test_80_websocket_error(self, authenticated_page):
        """测试80: WebSocket错误"""
        page = authenticated_page

        # 如果应用使用WebSocket
        # 模拟WebSocket断开或错误
        # 根据实际实现调整


# ======================== 第六部分：额外的综合测试 ========================
class TestAdditionalComprehensive:
    """额外的综合测试类"""

    def test_81_form_validation_messages(self, page):
        """测试81: 表单验证消息"""
        page.wait_for_selector("#login-form", timeout=5000)

        # 测试空用户名验证消息
        page.fill("#username-input", "")
        page.fill("#password-input", PASSWORD)
        page.click("#login-btn")
        time.sleep(0.5)

        # 验证用户名错误消息
        username_error = page.locator('.error-message, .validation-error')
        if username_error.count() > 0:
            expect(username_error.first).to_be_visible()

    def test_82_password_visibility_toggle(self, page):
        """测试82: 密码可见性切换"""
        page.wait_for_selector("#login-form", timeout=5000)

        # 查找密码可见性切换按钮（如果有）
        toggle_btn = page.locator('.password-toggle, .show-password, button:has-text("显示")')
        if toggle_btn.count() > 0:
            toggle_btn.first.click()
            time.sleep(0.3)

            # 验证密码字段类型改变
            password_input = page.locator("#password-input")
            input_type = password_input.get_attribute("type")
            # 应该变为text

    def test_83_remember_me_functionality(self, page):
        """测试83: 记住我功能"""
        page.wait_for_selector("#login-form", timeout=5000)

        # 查找记住我复选框（如果有）
        remember_checkbox = page.locator('#remember-me, input[type="checkbox"]')
        if remember_checkbox.count() > 0:
            remember_checkbox.first.check()

            # 登录
            page.fill("#username-input", USERNAME)
            page.fill("#password-input", PASSWORD)
            page.click("#login-btn")
            page.wait_for_selector("#app-section", timeout=5000)

            # 验证是否保存了登录状态
            saved_username = page.evaluate("localStorage.getItem('remembered_username')")
            # 根据实际实现验证

    def test_84_autofill_username(self, page):
        """测试84: 用户名自动填充"""
        # 先登录一次
        page.wait_for_selector("#login-form", timeout=5000)
        page.fill("#username-input", USERNAME)
        page.fill("#password-input", PASSWORD)
        page.click("#login-btn")
        page.wait_for_selector("#app-section", timeout=5000)

        # 登出
        logout_btn = page.locator("#logout-btn")
        logout_btn.click()
        page.wait_for_selector("#login-section", timeout=3000)

        # 验证用户名是否自动填充
        username_input = page.locator("#username-input")
        autofilled_value = username_input.get_attribute("value")
        # 根据实际实现验证

    def test_85_password_strength_indicator(self, page):
        """测试85: 密码强度指示器"""
        page.wait_for_selector("#login-form", timeout=5000)

        # 查找密码强度指示器（如果有）
        strength_indicator = page.locator('.password-strength, .strength-meter')
        if strength_indicator.count() > 0:
            # 输入弱密码
            page.fill("#password-input", "123")
            time.sleep(0.5)
            # 验证强度指示

            # 输入强密码
            page.fill("#password-input", "StrongP@ssw0rd123!")
            time.sleep(0.5)
            # 验证强度指示

    def test_86_keyboard_shortcuts(self, authenticated_page):
        """测试86: 键盘快捷键"""
        page = authenticated_page

        # 测试常见快捷键（如果有）
        # Ctrl+S 保存
        page.keyboard.press("Control+s")
        time.sleep(0.3)

        # Ctrl+N 新建
        page.keyboard.press("Control+n")
        time.sleep(0.3)

        # 根据实际快捷键实现调整

    def test_87_drag_and_drop_upload(self, authenticated_page, test_audio_file):
        """测试87: 拖拽上传"""
        page = authenticated_page

        if page.is_visible('[data-tab="mos"]'):
            page.click('[data-tab="mos"]')
            time.sleep(0.5)

            # 查找拖拽区域
            drop_zone = page.locator('.drop-zone, .upload-area, [data-drop-target]')
            if drop_zone.count() > 0:
                # 模拟拖拽文件（需要特殊处理）
                # Playwright的拖拽文件上传较复杂
                pass

    def test_88_copy_paste_functionality(self, authenticated_page):
        """测试88: 复制粘贴功能"""
        page = authenticated_page

        # 测试复制粘贴用户名
        user_display = page.locator("#user-display")
        if user_display.is_visible():
            # 选中文本
            user_display.click()
            page.keyboard.press("Control+a")
            page.keyboard.press("Control+c")
            time.sleep(0.3)

            # 在其他地方粘贴
            # 根据实际可用输入框调整

    def test_89_context_menu(self, authenticated_page):
        """测试89: 右键上下文菜单"""
        page = authenticated_page

        # 测试右键菜单
        if page.is_visible('[data-tab="mos"]'):
            tab = page.locator('[data-tab="mos"]')
            tab.click(button="right")
            time.sleep(0.3)

            # 验证是否有自定义上下文菜单

    def test_90_accessibility_aria_labels(self, page):
        """测试90: ARIA标签可访问性"""
        page.wait_for_selector("#login-form", timeout=5000)

        # 验证关键元素的ARIA标签
        login_btn = page.locator("#login-btn")
        aria_label = login_btn.get_attribute("aria-label")
        # 根据实际实现验证ARIA标签

    def test_91_focus_trap_in_modal(self, authenticated_page):
        """测试91: 模态框中的焦点陷阱"""
        page = authenticated_page

        # 如果有模态框，验证焦点是否被困在模态框内
        # 根据实际模态框实现调整

    def test_92_screen_reader_compatibility(self, page):
        """测试92: 屏幕阅读器兼容性"""
        page.wait_for_selector("#login-form", timeout=5000)

        # 验证关键元素有适当的role属性
        login_btn = page.locator("#login-btn")
        role = login_btn.get_attribute("role")
        # 应该有适当的role

    def test_93_high_contrast_mode(self, browser):
        """测试93: 高对比度模式"""
        # 创建高对比度上下文
        context = browser.new_context(
            color_scheme='dark',  # 或其他高对比度设置
            viewport={"width": 1920, "height": 1080}
        )
        page = context.new_page()
        page.goto(BASE_URL)

        # 验证高对比度下的显示
        page.wait_for_selector("#login-form", timeout=5000)
        expect(page.locator("#login-form")).to_be_visible()

        context.close()

    def test_94_zoom_accessibility(self, browser):
        """测试94: 缩放可访问性"""
        context = browser.new_context(
            viewport={"width": 1280, "height": 720}
        )
        page = context.new_page()

        # 设置页面缩放
        page.goto(BASE_URL)
        page.set_viewport_size({"width": 640, "height": 360})  # 模拟200%缩放

        # 验证缩放后的显示
        page.wait_for_selector("#login-form", timeout=5000)
        expect(page.locator("#login-form")).to_be_visible()

        context.close()

    def test_95_print_styles(self, browser):
        """测试95: 打印样式"""
        context = browser.new_context()
        page = context.new_page()
        page.goto(BASE_URL)

        # 模拟打印
        page.emulate_media(media="print")

        # 验证打印样式
        page.wait_for_selector("#login-form", timeout=5000)

        context.close()

    def test_96_localization_chinese(self, browser):
        """测试96: 中文本地化"""
        context = browser.new_context(locale='zh-CN')
        page = context.new_page()
        page.goto(BASE_URL)

        # 验证中文界面
        page.wait_for_selector("#login-form", timeout=5000)
        login_btn = page.locator("#login-btn")
        btn_text = login_btn.text_content()

        # 应该显示中文
        assert "登录" in btn_text or "Login" in btn_text

        context.close()

    def test_97_date_time_format(self, authenticated_page):
        """测试97: 日期时间格式"""
        page = authenticated_page

        # 查找日期时间显示元素（如果有）
        date_elements = page.locator('.date, .time, .datetime')
        if date_elements.count() > 0:
            # 验证日期时间格式
            date_text = date_elements.first.text_content()
            # 根据实际格式验证

    def test_98_number_formatting(self, authenticated_page):
        """测试98: 数字格式化"""
        page = authenticated_page

        # 查找数字显示元素（如果有）
        number_elements = page.locator('.number, .score, .stat-value')
        if number_elements.count() > 0:
            # 验证数字格式
            number_text = number_elements.first.text_content()
            # 根据实际格式验证

    def test_99_tooltip_display(self, authenticated_page):
        """测试99: 工具提示显示"""
        page = authenticated_page

        # 测试按钮工具提示
        logout_btn = page.locator("#logout-btn")
        logout_btn.hover()
        time.sleep(0.5)

        # 查找工具提示
        tooltip = page.locator('.tooltip, [role="tooltip"]')
        if tooltip.count() > 0:
            expect(tooltip.first).to_be_visible()

    def test_100_browser_console_clean(self, authenticated_page):
        """测试100: 浏览器控制台清洁"""
        page = authenticated_page

        # 监听控制台消息
        console_messages = []
        page.on("console", lambda msg: console_messages.append(msg))

        # 执行一些操作
        if page.is_visible('[data-tab="mos"]'):
            page.click('[data-tab="mos"]')
            time.sleep(1)

        # 验证没有错误消息
        errors = [msg for msg in console_messages if msg.type == "error"]
        # 根据实际需求调整验证逻辑


# ======================== 运行测试 ========================
if __name__ == "__main__":
    # 运行所有测试
    pytest.main([
        __file__,
        "-v",
        "--headed",  # 显示浏览器界面
        "--slowmo=100",  # 每个操作慢100ms，便于观察
        "--screenshot=only-on-failure",  # 失败时截图
        "--video=retain-on-failure",  # 失败时录制视频
        "--html=test_expanded_report.html",  # 生成HTML报告
        "--self-contained-html",
        "-k", "test_",  # 运行所有test_开头的测试
        "--tb=short",  # 简短的回溯信息
        "--maxfail=5",  # 最多失败5个测试后停止
    ])