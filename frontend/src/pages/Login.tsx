import React, { useState, useEffect } from 'react';
import { Form, Input, Button, Typography, message } from 'antd';
import { SoundOutlined, LockOutlined, UserOutlined, AudioOutlined, BarChartOutlined, CheckCircleOutlined, CustomerServiceFilled, PlayCircleFilled, PauseCircleFilled, SoundFilled } from '@ant-design/icons';
import { useAuth } from '../contexts/AuthContext';
import './Login.css';

const { Title, Text } = Typography;

const Login: React.FC = () => {
  const { login } = useAuth();
  const [loading, setLoading] = useState(false);
  const [mounted, setMounted] = useState(false);

  useEffect(() => {
    setMounted(true);
  }, []);

  const onFinish = async (values: { username: string; password: string }) => {
    setLoading(true);
    try {
      await login(values.username, values.password);
      message.success('登录成功');
      // 登录成功后，PublicRoute 会自动检测 isAuthenticated 并跳转到 /
    } catch (error: any) {
      const errMsg = error?.response?.data?.detail || error?.message || '登录失败';
      message.error(errMsg);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="login-container">
      {/* 背景动画元素 */}
      <div className="bg-shapes">
        <div className="shape shape-1"></div>
        <div className="shape shape-2"></div>
        <div className="shape shape-3"></div>
        <div className="shape shape-4"></div>
        <div className="shape shape-5"></div>
      </div>

      {/* 浮动音频图标 */}
      <div className="audio-icons">
        <div className="audio-icon icon-1"><AudioOutlined /></div>
        <div className="audio-icon icon-2"><CustomerServiceFilled /></div>
        <div className="audio-icon icon-3"><SoundFilled /></div>
        <div className="audio-icon icon-4"><SoundOutlined /></div>
        <div className="audio-icon icon-5"><PlayCircleFilled /></div>
        <div className="audio-icon icon-6"><PauseCircleFilled /></div>
        <div className="audio-icon icon-7"><SoundFilled /></div>
        <div className="audio-icon icon-8"><AudioOutlined /></div>
      </div>

      {/* 主内容区 */}
      <div className={`login-content ${mounted ? 'mounted' : ''}`}>
        {/* 左侧品牌展示 */}
        <div className="brand-section">
          <div className="brand-logo">
            <div className="logo-pulse">
              <SoundOutlined className="logo-icon" />
            </div>
          </div>
          <h1 className="brand-title">AudioMOS</h1>
          <p className="brand-subtitle">专业音频质量评分系统</p>
          
          <div className="feature-list">
            <div className="feature-item">
              <CheckCircleOutlined className="feature-icon" />
              <span>多维度MOS评分</span>
            </div>
            <div className="feature-item">
              <BarChartOutlined className="feature-icon" />
              <span>智能音频分析</span>
            </div>
            <div className="feature-item">
              <AudioOutlined className="feature-icon" />
              <span>批量处理支持</span>
            </div>
          </div>
        </div>

        {/* 右侧登录表单 */}
        <div className="login-card">
          <div className="login-header">
            <div className="login-icon-wrapper">
              <LockOutlined className="login-icon" />
            </div>
            <Title level={3} className="login-title">
              欢迎登录
            </Title>
            <Text type="secondary" className="login-subtitle">
              请输入您的账号信息
            </Text>
          </div>

          <Form
            name="login"
            onFinish={onFinish}
            autoComplete="off"
            size="large"
            initialValues={{ username: 'admin' }}
            className="login-form"
          >
            <Form.Item
              name="username"
              rules={[{ required: true, message: '请输入用户名' }]}
            >
              <Input
                prefix={<UserOutlined className="input-icon" />}
                placeholder="用户名"
                className="login-input"
              />
            </Form.Item>

            <Form.Item
              name="password"
              rules={[{ required: true, message: '请输入密码' }]}
            >
              <Input.Password
                prefix={<LockOutlined className="input-icon" />}
                placeholder="密码"
                className="login-input"
              />
            </Form.Item>

            <Form.Item className="login-button-wrapper">
              <Button
                type="primary"
                htmlType="submit"
                loading={loading}
                block
                className="login-button"
              >
                <span className="button-text">登录</span>
                <div className="button-shine"></div>
              </Button>
            </Form.Item>
          </Form>

          <div className="login-tips">
            <Text type="secondary" style={{ fontSize: '12px' }}>
              首次登录请联系管理员获取账号
            </Text>
          </div>
        </div>
      </div>

      {/* 底部版权 */}
      <div className="login-footer">
        <Text type="secondary">
          © {new Date().getFullYear()} AudioMOS - 音频质量评分系统
        </Text>
      </div>
    </div>
  );
};

export default Login;
