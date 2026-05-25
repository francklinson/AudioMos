import React, { useState, useEffect } from 'react';
import {
  Layout,
  Typography,
  Row,
  Col,
  Card,
  Upload,
  Button,
  Table,
  Tag,
  Progress,
  Space,
  message,
  Modal,
  Statistic,
  Empty,
  Spin,
  Drawer,
  Checkbox,
  Collapse,
  Tooltip
} from 'antd';
import {
  SoundOutlined,
  UploadOutlined,
  PlayCircleOutlined,
  DownloadOutlined,
  DeleteOutlined,
  FileExcelOutlined,
  CheckCircleOutlined,
  CloseCircleOutlined,
  SyncOutlined,
  BarChartOutlined,
  InfoCircleOutlined,
  LogoutOutlined,
  EyeOutlined,
  SettingOutlined
} from '@ant-design/icons';
import type { UploadFile, UploadProps } from 'antd/es/upload';
import { useAuth } from '../contexts/AuthContext';
import { mosApi } from '../services/api';
import dayjs from 'dayjs';
import './Home.css';

const { Header, Content, Footer } = Layout;
const { Title, Text } = Typography;
const { Panel } = Collapse;

// 计算项目配置选项
interface MetricConfig {
  key: string;
  label: string;
  description: string;
  category: 'ref' | 'no_ref';
  defaultChecked: boolean;
}

const METRIC_OPTIONS: MetricConfig[] = [
  // 有参考指标
  { key: 'pesq', label: 'PESQ', description: '语音质量感知评估', category: 'ref', defaultChecked: true },
  { key: 'stoi', label: 'STOI', description: '短时客观可懂度', category: 'ref', defaultChecked: true },
  { key: 'sisdr', label: 'SISDR', description: '尺度不变信噪比', category: 'ref', defaultChecked: true },
  { key: 'wer', label: 'WER', description: '词错误率', category: 'ref', defaultChecked: true },
  { key: 'tcf', label: '音色还原度', description: '基于说话人验证模型的音色相似度', category: 'ref', defaultChecked: true },
  // 无参考指标
  { key: 'dnsmos', label: 'DNSMOS', description: '深度噪声抑制MOS评分', category: 'no_ref', defaultChecked: true },
  { key: 'nisqa', label: 'NISQA', description: '语音质量神经网络评估', category: 'no_ref', defaultChecked: true },
  { key: 'scoreq', label: 'Scoreq', description: '基于深度学习的语音质量评估', category: 'no_ref', defaultChecked: true },
  { key: 'utmos', label: 'UTMOS', description: 'UTokyo-SaruLab MOS预测系统', category: 'no_ref', defaultChecked: true },
];

interface Task {
  task_id: string;
  status: 'pending' | 'queued' | 'processing' | 'completed' | 'failed';
  progress: number;
  message: string;
  result_file?: string;
  created_at: string;
  updated_at: string;
  uploaded_files?: string[];
}

interface TaskResult {
  task_id: string;
  status: string;
  results: Record<string, any>[];
  columns: string[];
  total_files: number;
}

const Home: React.FC = () => {
  const { user, logout } = useAuth();
  const [fileList, setFileList] = useState<UploadFile[]>([]);
  const [uploading, setUploading] = useState(false);
  const [tasks, setTasks] = useState<Task[]>([]);

  // 结果展示相关状态
  const [resultDrawerVisible, setResultDrawerVisible] = useState(false);
  const [selectedTaskResult, setSelectedTaskResult] = useState<TaskResult | null>(null);
  const [resultLoading, setResultLoading] = useState(false);
  // 结果表格分页状态
  const [resultPagination, setResultPagination] = useState({
    current: 1,
    pageSize: 10,
  });

  // 计算项目配置状态
  const [selectedMetrics, setSelectedMetrics] = useState<string[]>(
    METRIC_OPTIONS.filter(m => m.defaultChecked).map(m => m.key)
  );
  const [configPanelVisible, setConfigPanelVisible] = useState(false);

  // 加载任务列表
  const loadTasks = async () => {
    try {
      const data = await mosApi.getTasks();
      setTasks(data);
    } catch (error) {
      console.error('加载任务失败:', error);
    }
  };

  useEffect(() => {
    loadTasks();
    const interval = setInterval(loadTasks, 5000);
    return () => clearInterval(interval);
  }, []);

  // 同步任务列表中的处理中任务到currentTask
  useEffect(() => {
    const processingTask = tasks.find(t => t.status === 'processing');
    if (processingTask) {
      setCurrentTask(prev => {
        // 如果当前没有任务，或者任务ID相同但进度不同，则更新
        if (!prev || (prev.task_id === processingTask.task_id && prev.progress !== processingTask.progress)) {
          return processingTask;
        }
        return prev;
      });
    }
  }, [tasks]);

  // WebSocket连接
  useEffect(() => {
    if (currentTask?.task_id && currentTask.status === 'processing') {
      const wsUrl = `ws://localhost:8000/api/mos/ws/${currentTask.task_id}`;
      const socket = new WebSocket(wsUrl);

      socket.onmessage = (event) => {
        const data = JSON.parse(event.data);
        setCurrentTask(prev => prev ? { ...prev, ...data } : null);
      };

      socket.onerror = (error) => {
        console.error('WebSocket error:', error);
      };

      socket.onclose = () => {
        console.log('WebSocket closed');
      };

      return () => {
        socket.close();
      };
    }
  }, [currentTask?.task_id, currentTask?.status]);

  const handleUpload = async () => {
    if (fileList.length === 0) {
      message.warning('请选择要上传的文件');
      return;
    }

    const files: File[] = [];
    for (const f of fileList) {
      const file = f.originFileObj || f;
      if (file instanceof File) {
        files.push(file);
      }
    }

    const validFiles = files.filter(f =>
      f.name.toLowerCase().endsWith('.wav') || f.name.toLowerCase().endsWith('.mp3')
    );

    if (validFiles.length === 0) {
      message.error('请上传 .wav 或 .mp3 格式的音频文件');
      return;
    }

    if (validFiles.length !== files.length) {
      const invalidCount = files.length - validFiles.length;
      message.warning(`已过滤 ${invalidCount} 个非音频文件`);
    }

    setUploading(true);
    try {
      const data = await mosApi.uploadFiles(validFiles, selectedMetrics);
      message.success(data.message);
      setFileList([]);

      // 自动开始处理
      const processData = await mosApi.startProcess(data.task_id);
      message.success(`任务已提交到队列，排队位置: ${processData.queue_position || 1}`);

      // 清空文件列表，允许继续提交新任务
      setFileList([]);

      // 刷新任务列表
      await loadTasks();

      // 设置当前任务
      const newTask: Task = {
        task_id: data.task_id,
        status: 'queued',
        progress: 0,
        message: `已加入队列，排队位置: ${processData.queue_position || 1}`,
        created_at: new Date().toISOString(),
        updated_at: new Date().toISOString(),
        uploaded_files: data.files,
      };
      setCurrentTask(newTask);
    } catch (error: any) {
      message.error(error.response?.data?.detail || '上传失败');
    } finally {
      setUploading(false);
    }
  };

  const handleDownload = async (taskId: string) => {
    try {
      const blob = await mosApi.downloadResult(taskId);
      const url = window.URL.createObjectURL(new Blob([blob]));
      const link = document.createElement('a');
      link.href = url;
      link.setAttribute('download', `MOS评分结果_${taskId.slice(0, 8)}.xlsx`);
      document.body.appendChild(link);
      link.click();
      link.remove();
      window.URL.revokeObjectURL(url);
    } catch (error) {
      message.error('下载失败');
    }
  };

  const handleViewResults = async (taskId: string) => {
    setResultLoading(true);
    // 重置分页状态
    setResultPagination({ current: 1, pageSize: 10 });
    try {
      const data = await mosApi.getTaskResults(taskId);
      setSelectedTaskResult(data);
      setResultDrawerVisible(true);
    } catch (error: any) {
      message.error(error.response?.data?.detail || '获取结果失败');
    } finally {
      setResultLoading(false);
    }
  };

  const handleDelete = async (taskId: string) => {
    Modal.confirm({
      title: '确认删除',
      content: '删除后将无法恢复,是否确认?',
      onOk: async () => {
        try {
          await mosApi.deleteTask(taskId);
          message.success('删除成功');
          loadTasks();
          if (currentTask?.task_id === taskId) {
            setCurrentTask(null);
          }
        } catch (error) {
          message.error('删除失败');
        }
      },
    });
  };

  const handleLogout = async () => {
    await logout();
    message.success('已退出登录');
  };

  const uploadProps: UploadProps = {
    onRemove: (file) => {
      const index = fileList.indexOf(file);
      const newFileList = fileList.slice();
      newFileList.splice(index, 1);
      setFileList(newFileList);
    },
    beforeUpload: () => {
      // 阻止自动上传，改为手动控制
      return false;
    },
    onChange: (info) => {
      // 使用 onChange 来更新文件列表，确保多文件上传时能正确捕获所有文件
      const newFileList = info.fileList.filter((f) => {
        // 只保留状态为 uploading 或 done 的文件（即用户选择的文件）
        return f.status === 'uploading' || f.status === 'done' || !f.status;
      });
      setFileList(newFileList);
    },
    fileList,
    multiple: true,
    accept: '.wav,.mp3',
  };

  const getStatusTag = (status: string) => {
    switch (status) {
      case 'pending':
        return <Tag icon={<SyncOutlined spin />} color="processing">等待中</Tag>;
      case 'queued':
        return <Tag icon={<SyncOutlined spin />} color="orange">队列中</Tag>;
      case 'processing':
        return <Tag icon={<PlayCircleOutlined />} color="blue">处理中</Tag>;
      case 'completed':
        return <Tag icon={<CheckCircleOutlined />} color="success">已完成</Tag>;
      case 'failed':
        return <Tag icon={<CloseCircleOutlined />} color="error">失败</Tag>;
      default:
        return <Tag>未知</Tag>;
    }
  };

  const columns = [
    {
      title: '任务ID',
      dataIndex: 'task_id',
      key: 'task_id',
      render: (id: string) => id.slice(0, 8) + '...',
    },
    {
      title: '状态',
      dataIndex: 'status',
      key: 'status',
      render: getStatusTag,
    },
    {
      title: '进度',
      dataIndex: 'progress',
      key: 'progress',
      render: (progress: number, record: Task) => (
        record.status === 'processing' ? (
          <Progress percent={progress} size="small" />
        ) : (
          <Progress percent={progress} size="small" status={record.status === 'failed' ? 'exception' : 'success'} />
        )
      ),
    },
    {
      title: '创建时间',
      dataIndex: 'created_at',
      key: 'created_at',
      render: (date: string) => dayjs(date).format('MM-DD HH:mm'),
    },
    {
      title: '操作',
      key: 'action',
      render: (_: any, record: Task) => (
        <Space>
          {record.status === 'completed' && (
            <>
              <Button
                icon={<EyeOutlined />}
                size="small"
                onClick={() => handleViewResults(record.task_id)}
              >
                查看结果
              </Button>
              <Button
                type="primary"
                icon={<DownloadOutlined />}
                size="small"
                onClick={() => handleDownload(record.task_id)}
              >
                下载结果
              </Button>
            </>
          )}
          <Button
            danger
            icon={<DeleteOutlined />}
            size="small"
            onClick={() => handleDelete(record.task_id)}
          >
            删除
          </Button>
        </Space>
      ),
    },
  ];

  return (
    <Layout className="home-container" style={{ minHeight: '100vh' }}>
      {/* 动态背景 */}
      <div className="home-background">
        {/* 渐变圆形 */}
        <div className="bg-gradient-circle circle-1"></div>
        <div className="bg-gradient-circle circle-2"></div>
        <div className="bg-gradient-circle circle-3"></div>
        <div className="bg-gradient-circle circle-4"></div>
        <div className="bg-gradient-circle circle-5"></div>

        {/* 浮动装饰元素 */}
        <div className="floating-elements">
          <div className="float-item float-1"></div>
          <div className="float-item float-2"></div>
          <div className="float-item float-3"></div>
          <div className="float-item float-4"></div>
          <div className="float-item float-5"></div>
        </div>

        {/* 网格背景 */}
        <div className="grid-pattern"></div>

        {/* 粒子效果 */}
        <div className="particles">
          {[...Array(10)].map((_, i) => (
            <div key={i} className="particle"></div>
          ))}
        </div>
      </div>

      <Header style={{
        background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
        padding: '0 24px',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between'
      }}>
        <Space>
          <SoundOutlined style={{ fontSize: 28, color: '#fff' }} />
          <Title level={4} style={{ color: '#fff', margin: 0 }}>
            AudioMOS 音频质量评分系统
          </Title>
        </Space>
        <Space>
          <Text style={{ color: 'rgba(255,255,255,0.8)' }}>
            欢迎, {user?.username}
          </Text>
          <Button
            type="text"
            icon={<LogoutOutlined />}
            onClick={handleLogout}
            style={{ color: '#fff' }}
          >
            退出
          </Button>
        </Space>
      </Header>

      <Content className="home-content" style={{ padding: 24, background: 'transparent' }}>
        <div style={{ maxWidth: 1200, margin: '0 auto' }}>
        <Row gutter={[24, 24]}>
          {/* 第一行：上传区域 */}
          <Col span={24}>
            <Card
              className="home-card"
              title={
                <Space>
                  <UploadOutlined />
                  <span>上传音频文件</span>
                </Space>
              }
              variant="borderless"
              style={{ borderRadius: 12 }}
            >
              <div style={{
                marginBottom: 16,
                padding: '16px 20px',
                background: 'linear-gradient(135deg, #f0f5ff 0%, #e6f0ff 100%)',
                borderRadius: 12,
                border: '1px solid #d6e4ff',
                display: 'flex',
                alignItems: 'flex-start',
                gap: 12
              }}>
                <InfoCircleOutlined style={{ fontSize: 20, color: '#667eea', marginTop: 2 }} />
                <div>
                  <div style={{ fontWeight: 600, color: '#1d39c4', marginBottom: 6, fontSize: 15 }}>
                    使用说明
                  </div>
                  <div style={{ color: '#4c5b8a', lineHeight: 1.6, fontSize: 14 }}>
                    支持上传 <Tag color="blue" style={{ margin: '0 4px' }}>.wav</Tag> 和 <Tag color="blue" style={{ margin: '0 4px' }}>.mp3</Tag> 格式的音频文件
                    <br />
                    系统会自动进行音频切分、对齐并计算多种MOS评分指标
                  </div>
                </div>
              </div>

              <Upload.Dragger {...uploadProps} style={{ marginBottom: 16 }}>
                <p className="ant-upload-drag-icon">
                  <SoundOutlined style={{ color: '#667eea' }} />
                </p>
                <p className="ant-upload-text">点击或拖拽文件到此处上传</p>
                <p className="ant-upload-hint">
                  支持单个或批量上传,文件格式: .wav, .mp3
                </p>
              </Upload.Dragger>

              {/* 计算项目配置面板 */}
              <Card
                size="small"
                style={{ marginBottom: 16, background: '#fafafa' }}
                title={
                  <Space>
                    <SettingOutlined />
                    <span>计算项目配置</span>
                    <Tag color="blue">{selectedMetrics.length} 项已选</Tag>
                  </Space>
                }
                extra={
                  <Button
                    type="link"
                    size="small"
                    onClick={() => setConfigPanelVisible(!configPanelVisible)}
                  >
                    {configPanelVisible ? '收起' : '展开'}
                  </Button>
                }
              >
                {configPanelVisible && (
                  <div>
                    <div style={{ marginBottom: 12 }}>
                      <Space>
                        <Button
                          size="small"
                          onClick={() => setSelectedMetrics(METRIC_OPTIONS.map(m => m.key))}
                        >
                          全选
                        </Button>
                        <Button
                          size="small"
                          onClick={() => setSelectedMetrics([])}
                        >
                          全不选
                        </Button>
                        <Button
                          size="small"
                          onClick={() => setSelectedMetrics(METRIC_OPTIONS.filter(m => m.defaultChecked).map(m => m.key))}
                        >
                          恢复默认
                        </Button>
                      </Space>
                    </div>

                    <Checkbox.Group
                      value={selectedMetrics}
                      onChange={(values) => setSelectedMetrics(values as string[])}
                    >
                      <Collapse ghost defaultActiveKey={['ref', 'no_ref']}>
                        <Panel header={<Text strong>有参考音频指标 (需要参考音频)</Text>} key="ref">
                          <Row gutter={[16, 8]}>
                            {METRIC_OPTIONS.filter(m => m.category === 'ref').map(metric => (
                              <Col span={12} key={metric.key}>
                                <Tooltip title={metric.description}>
                                  <Checkbox value={metric.key}>
                                    <Text strong>{metric.label}</Text>
                                    <br />
                                    <Text type="secondary" style={{ fontSize: 12 }}>{metric.description}</Text>
                                  </Checkbox>
                                </Tooltip>
                              </Col>
                            ))}
                          </Row>
                        </Panel>

                        <Panel header={<Text strong>无参考音频指标 (无需参考音频)</Text>} key="no_ref">
                          <Row gutter={[16, 8]}>
                            {METRIC_OPTIONS.filter(m => m.category === 'no_ref').map(metric => (
                              <Col span={12} key={metric.key}>
                                <Tooltip title={metric.description}>
                                  <Checkbox value={metric.key}>
                                    <Text strong>{metric.label}</Text>
                                    <br />
                                    <Text type="secondary" style={{ fontSize: 12 }}>{metric.description}</Text>
                                  </Checkbox>
                                </Tooltip>
                              </Col>
                            ))}
                          </Row>
                        </Panel>
                      </Collapse>
                    </Checkbox.Group>
                  </div>
                )}

                {!configPanelVisible && (
                  <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8 }}>
                    {selectedMetrics.map(key => {
                      const metric = METRIC_OPTIONS.find(m => m.key === key);
                      return metric ? (
                        <Tag key={key} color="blue">{metric.label}</Tag>
                      ) : null;
                    })}
                  </div>
                )}
              </Card>

              <Button
                type="primary"
                onClick={handleUpload}
                loading={uploading}
                disabled={fileList.length === 0 || selectedMetrics.length === 0}
                block
                size="large"
                icon={uploading ? <SyncOutlined spin /> : <UploadOutlined />}
                style={{
                  background: fileList.length === 0 || selectedMetrics.length === 0
                    ? 'linear-gradient(135deg, #d9d9d9 0%, #bfbfbf 100%)'
                    : 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
                  border: 'none',
                  borderRadius: 10,
                  height: 48,
                  fontSize: 16,
                  fontWeight: 500,
                  boxShadow: fileList.length === 0 || selectedMetrics.length === 0
                    ? 'none'
                    : '0 4px 15px rgba(102, 126, 234, 0.4)',
                  transition: 'all 0.3s ease'
                }}
              >
                {uploading ? '上传处理中...' : fileList.length === 0 ? '请选择音频文件' : selectedMetrics.length === 0 ? '请至少选择一项计算指标' : `开始上传并处理 (${fileList.length} 个文件)`}
              </Button>


            </Card>
          </Col>

          {/* 第二行：统计信息 */}
          <Col span={24}>
            <Row gutter={[16, 16]}>
              <Col xs={24} md={8}>
                <Card className="home-card stat-card" variant="borderless" style={{ borderRadius: 12 }}>
                  <Statistic
                    title="总任务数"
                    value={tasks.length}
                    prefix={<BarChartOutlined />}
                  />
                </Card>
              </Col>
              <Col xs={24} md={8}>
                <Card className="home-card stat-card" variant="borderless" style={{ borderRadius: 12 }}>
                  <Statistic
                    title="已完成"
                    value={tasks.filter(t => t.status === 'completed').length}
                    valueStyle={{ color: '#3f8600' }}
                    prefix={<CheckCircleOutlined />}
                  />
                </Card>
              </Col>
              <Col xs={24} md={8}>
                <Card className="home-card stat-card" variant="borderless" style={{ borderRadius: 12 }}>
                  <Statistic
                    title="处理中"
                    value={tasks.filter(t => t.status === 'processing').length}
                    valueStyle={{ color: '#1890ff' }}
                    prefix={<SyncOutlined spin />}
                  />
                </Card>
              </Col>
            </Row>
          </Col>

          {/* 第三行：任务列表 */}
          <Col span={24}>
            <Card
              className="home-card"
              title={
                <Space>
                  <FileExcelOutlined />
                  <span>任务列表</span>
                </Space>
              }
              variant="borderless"
              style={{ borderRadius: 12 }}
              extra={
                <Button onClick={loadTasks} icon={<SyncOutlined />}>
                  刷新
                </Button>
              }
            >
              {tasks.length === 0 ? (
                <Empty
                  image={Empty.PRESENTED_IMAGE_SIMPLE}
                  description="暂无任务"
                />
              ) : (
                <Table
                  dataSource={tasks}
                  columns={columns}
                  rowKey="task_id"
                  size="small"
                  pagination={{
                    pageSize: 5,
                    showSizeChanger: true,
                    pageSizeOptions: [5, 10, 20, 50],
                    showTotal: (total, range) => `第 ${range[0]}-${range[1]} 条 / 共 ${total} 条`,
                  }}
                />
              )}
            </Card>
          </Col>
        </Row>
        </div>
      </Content>

      <Footer style={{ textAlign: 'center', background: 'transparent' }}>
        <Text type="secondary">
          AudioMOS ©{new Date().getFullYear()} | 音频质量评分系统
        </Text>
      </Footer>

      {/* 结果展示抽屉 */}
      <Drawer
        title="MOS评分结果详情"
        placement="right"
        width={1000}
        onClose={() => setResultDrawerVisible(false)}
        open={resultDrawerVisible}
      >
        {resultLoading ? (
          <div style={{ textAlign: 'center', padding: '50px' }}>
            <Spin size="large" />
            <p style={{ marginTop: 16 }}>正在加载结果...</p>
          </div>
        ) : selectedTaskResult ? (
          <div>
            <Card className="home-card" style={{ marginBottom: 16 }}>
              <Space direction="vertical" style={{ width: '100%' }}>
                <Text strong>任务ID: {selectedTaskResult.task_id}</Text>
                <Text>文件数量: {selectedTaskResult.total_files} 个</Text>
              </Space>
            </Card>

            <Table
              dataSource={selectedTaskResult.results}
              rowKey={(record) => `${record['文件名'] || record['file'] || 'file'}_${record['任务ID'] || Math.random().toString(36).substr(2, 9)}`}
              columns={selectedTaskResult.columns.map((col, idx) => ({
                title: col,
                dataIndex: col,
                key: `${col}_${idx}`,
                render: (value: any) => {
                  if (typeof value === 'number') {
                    return value.toFixed(4);
                  }
                  return value;
                },
              }))}
              scroll={{ x: 'max-content' }}
              pagination={{
                current: resultPagination.current,
                pageSize: resultPagination.pageSize,
                showSizeChanger: true,
                pageSizeOptions: [10, 20, 50, 100],
                showTotal: (total, range) => `第 ${range[0]}-${range[1]} 条 / 共 ${total} 条`,
                onChange: (page, pageSize) => {
                  setResultPagination({ current: page, pageSize: pageSize || 10 });
                },
                onShowSizeChange: (current, size) => {
                  setResultPagination({ current: 1, pageSize: size });
                },
              }}
              size="small"
              bordered
            />
          </div>
        ) : (
          <Empty description="暂无结果数据" />
        )}
      </Drawer>
    </Layout>
  );
};

export default Home;
