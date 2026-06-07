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
  Tooltip,
  Tabs,
  Select
} from 'antd';
import {
  SoundOutlined,
  UploadOutlined,
  PlayCircleOutlined,
  PauseCircleOutlined,
  DownloadOutlined,
  DeleteOutlined,
  FileExcelOutlined,
  CheckCircleOutlined,
  CloseCircleOutlined,
  SyncOutlined,
  BarChartOutlined,
  LogoutOutlined,
  EyeOutlined,
  SettingOutlined,
  ToolOutlined
} from '@ant-design/icons';
import type { UploadFile, UploadProps } from 'antd/es/upload';
import { useAuth } from '../contexts/AuthContext';
import { mosApi, restorationApi } from '../services/api';
import AudioComparison from '../components/AudioComparison';
import dayjs from 'dayjs';
import './Home.css';

const { Header, Content, Footer } = Layout;
const { Title, Text } = Typography;
const { Panel } = Collapse;

const METRIC_OPTIONS = [
  { key: 'pesq', label: 'PESQ', description: '语音质量感知评估', category: 'ref', defaultChecked: true },
  { key: 'stoi', label: 'STOI', description: '短时客观可懂度', category: 'ref', defaultChecked: true },
  { key: 'sisdr', label: 'SISDR', description: '尺度不变信噪比', category: 'ref', defaultChecked: true },
  { key: 'wer', label: 'WER', description: '词错误率', category: 'ref', defaultChecked: true },
  { key: 'tcf', label: '音色还原度', description: '基于说话人验证模型的音色相似度', category: 'ref', defaultChecked: true },
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

interface RestorationAlgorithm {
  name: string;
  display_name: string;
  description: string;
  type: string;
  advantages: string[];
  limitations: string[];
  initialized: boolean;
}

interface RestorationTask {
  task_id: string;
  algorithm: string;
  filename: string;
  status: string;
  created_at: string;
  progress: number;
  message: string;
  result_file: string | null;
  processing_time: number | null;
  metadata: Record<string, any> | null;
}

const Home: React.FC = () => {
  const { user, logout } = useAuth();
  const [activeTab, setActiveTab] = useState('mos');

  const [fileList, setFileList] = useState<UploadFile[]>([]);
  const [uploading, setUploading] = useState(false);
  const [tasks, setTasks] = useState<Task[]>([]);
  const [resultDrawerVisible, setResultDrawerVisible] = useState(false);
  const [selectedTaskResult, setSelectedTaskResult] = useState<TaskResult | null>(null);
  const [resultLoading, setResultLoading] = useState(false);
  const [resultPagination, setResultPagination] = useState({ current: 1, pageSize: 10 });
  const [taskPagination, setTaskPagination] = useState({ current: 1, pageSize: 5 });
  const [selectedMetrics, setSelectedMetrics] = useState<string[]>(
    METRIC_OPTIONS.filter(m => m.defaultChecked).map(m => m.key)
  );
  const [configPanelVisible, setConfigPanelVisible] = useState(false);

  const [restorationAlgorithms, setRestorationAlgorithms] = useState<RestorationAlgorithm[]>([]);
  const [selectedRestorationAlgorithm, setSelectedRestorationAlgorithm] = useState('');
  const [restorationFiles, setRestorationFiles] = useState<File[]>([]);
  const [restorationTasks, setRestorationTasks] = useState<RestorationTask[]>([]);
  const [restorationLoading, setRestorationLoading] = useState(false);
  const [expandedRestorationTask, setExpandedRestorationTask] = useState<string | null>(null);

  const loadTasks = async () => {
    try {
      const data = await mosApi.getTasks();
      setTasks(data);
    } catch (error) {
      console.error('加载任务失败:', error);
    }
  };

  const loadRestorationAlgorithms = async () => {
    try {
      const data = await restorationApi.getAlgorithms();
      setRestorationAlgorithms(data);
      if (data.length > 0 && !selectedRestorationAlgorithm) {
        const preferred = data.find((a: RestorationAlgorithm) => a.name === 'clearvoice_frcrn_se_16k');
        setSelectedRestorationAlgorithm(preferred ? preferred.name : data[0].name);
      }
    } catch (error) {
      console.error('加载修复算法失败:', error);
    }
  };

  const loadRestorationTasks = async () => {
    try {
      const data = await restorationApi.getTasks();
      setRestorationTasks(data || []);
    } catch (error) {
      console.error('加载修复任务失败:', error);
    }
  };

  useEffect(() => {
    loadTasks();
    loadRestorationAlgorithms();
    loadRestorationTasks();

    const interval = setInterval(() => {
      loadTasks();
      loadRestorationTasks();
    }, 5000);
    return () => clearInterval(interval);
  }, []);

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

      const processData = await mosApi.startProcess(data.task_id);
      message.success(`任务已提交到队列，排队位置: ${processData.queue_position || 1}`);

      setFileList([]);
      await loadTasks();
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
    beforeUpload: () => false,
    onChange: (info) => {
      const newFileList = info.fileList.filter((f) => {
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

  const handleRestorationSubmit = async () => {
    if (restorationFiles.length === 0) {
      message.error('请选择音频文件');
      return;
    }
    if (!selectedRestorationAlgorithm) {
      message.error('请选择修复算法');
      return;
    }

    setRestorationLoading(true);

    try {
      const uploadResult = await restorationApi.uploadBatch(restorationFiles, selectedRestorationAlgorithm);
      // 逐个提交处理
      for (const taskId of uploadResult.task_ids) {
        await restorationApi.processTask(taskId);
      }
      message.success(`已提交 ${uploadResult.count} 个文件，正在处理...`);
      setRestorationFiles([]);
      loadRestorationTasks();
    } catch (err: any) {
      message.error(err.response?.data?.detail || '操作失败');
    } finally {
      setRestorationLoading(false);
    }
  };

  const handleRestorationDownload = async (taskId: string) => {
    try {
      await restorationApi.downloadResult(taskId);
      message.success('下载成功');
    } catch (err: any) {
      message.error('下载失败');
    }
  };

  const handleRestorationDelete = async (taskId: string) => {
    if (!window.confirm('确定要删除该任务吗？')) return;
    try {
      await restorationApi.deleteTask(taskId);
      loadRestorationTasks();
      message.success('删除成功');
    } catch (err: any) {
      message.error('删除失败');
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

  const restorationColumns = [
    {
      title: '文件名',
      dataIndex: 'filename',
      key: 'filename',
    },
    {
      title: '算法',
      dataIndex: 'algorithm',
      key: 'algorithm',
      render: (algo: string) => getRestorationAlgoLabel(algo),
    },
    {
      title: '状态',
      dataIndex: 'status',
      key: 'status',
      render: getStatusTag,
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
      render: (_: any, record: RestorationTask) => (
        <Space>
          {record.status === 'completed' && (
            <>
              <Button
                type="link"
                size="small"
                icon={expandedRestorationTask === record.task_id ? <PauseCircleOutlined /> : <PlayCircleOutlined />}
                onClick={() => setExpandedRestorationTask(
                  expandedRestorationTask === record.task_id ? null : record.task_id
                )}
                style={{ color: '#4fc3f7' }}
              >
                {expandedRestorationTask === record.task_id ? '收起对比' : '试听对比'}
              </Button>
              <Button
                type="primary"
                icon={<DownloadOutlined />}
                size="small"
                onClick={() => handleRestorationDownload(record.task_id)}
              >
                下载
              </Button>
            </>
          )}
          <Button
            danger
            icon={<DeleteOutlined />}
            size="small"
            onClick={() => handleRestorationDelete(record.task_id)}
          >
            删除
          </Button>
        </Space>
      ),
    },
  ];

  const renderMosTab = () => (
    <div>
      <Row gutter={[24, 24]}>
        <Col span={24}>
          <Card
            className="home-card"
            title={<Space><UploadOutlined /><span>上传音频文件</span></Space>}
            variant="borderless"
            style={{ borderRadius: 12 }}
          >
            <Text type="secondary" style={{ fontSize: 13, marginBottom: 16, display: 'block' }}>
              💡 支持 .wav / .mp3 格式，自动切分对齐并计算多种 MOS 评分指标
            </Text>

            <Upload.Dragger {...uploadProps} style={{ marginBottom: 16 }}>
              <p className="ant-upload-drag-icon">
                <SoundOutlined style={{ color: '#667eea' }} />
              </p>
              <p className="ant-upload-text">点击或拖拽文件到此处上传</p>
              <p className="ant-upload-hint">支持单个或批量上传,文件格式: .wav, .mp3</p>
            </Upload.Dragger>

            <Card
              size="small"
              style={{ marginBottom: 16, background: '#fafafa' }}
              title={<Space><SettingOutlined /><span>计算项目配置</span><Tag color="blue">{selectedMetrics.length} 项已选</Tag></Space>}
              extra={<Button type="link" size="small" onClick={() => setConfigPanelVisible(!configPanelVisible)}>{configPanelVisible ? '收起' : '展开'}</Button>}
            >
              {configPanelVisible && (
                <div>
                  <div style={{ marginBottom: 12 }}>
                    <Space>
                      <Button size="small" onClick={() => setSelectedMetrics(METRIC_OPTIONS.map(m => m.key))}>全选</Button>
                      <Button size="small" onClick={() => setSelectedMetrics([])}>全不选</Button>
                      <Button size="small" onClick={() => setSelectedMetrics(METRIC_OPTIONS.filter(m => m.defaultChecked).map(m => m.key))}>恢复默认</Button>
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
                    return metric ? <Tag key={key} color="blue">{metric.label}</Tag> : null;
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
              }}
            >
              {uploading ? '上传处理中...' : fileList.length === 0 ? '请选择音频文件' : selectedMetrics.length === 0 ? '请至少选择一项计算指标' : `开始上传并处理 (${fileList.length} 个文件)`}
            </Button>
          </Card>
        </Col>

        <Col span={24}>
          <Row gutter={[16, 16]}>
            <Col xs={24} md={8}>
              <Card className="home-card stat-card" variant="borderless" style={{ borderRadius: 12 }}>
                <Statistic title="总任务数" value={tasks.length} prefix={<BarChartOutlined />} />
              </Card>
            </Col>
            <Col xs={24} md={8}>
              <Card className="home-card stat-card" variant="borderless" style={{ borderRadius: 12 }}>
                <Statistic title="已完成" value={tasks.filter(t => t.status === 'completed').length} valueStyle={{ color: '#3f8600' }} prefix={<CheckCircleOutlined />} />
              </Card>
            </Col>
            <Col xs={24} md={8}>
              <Card className="home-card stat-card" variant="borderless" style={{ borderRadius: 12 }}>
                <Statistic title="处理中" value={tasks.filter(t => t.status === 'processing').length} valueStyle={{ color: '#1890ff' }} prefix={<SyncOutlined spin />} />
              </Card>
            </Col>
          </Row>
        </Col>

        <Col span={24}>
          <Card
            className="home-card"
            title={<Space><FileExcelOutlined /><span>任务列表</span></Space>}
            variant="borderless"
            style={{ borderRadius: 12 }}
            extra={<Button onClick={loadTasks} icon={<SyncOutlined />}>刷新</Button>}
          >
            {tasks.length === 0 ? (
              <Empty image={Empty.PRESENTED_IMAGE_SIMPLE} description="暂无任务" />
            ) : (
              <Table
                dataSource={tasks}
                columns={columns}
                rowKey="task_id"
                size="small"
                pagination={{
                  current: taskPagination.current,
                  pageSize: taskPagination.pageSize,
                  showSizeChanger: true,
                  pageSizeOptions: [5, 10, 20, 50],
                  showTotal: (total, range) => `第 ${range[0]}-${range[1]} 条 / 共 ${total} 条`,
                  onChange: (page, pageSize) => {
                    setTaskPagination({ current: page, pageSize: pageSize || 5 });
                  },
                  onShowSizeChange: (_current, size) => {
                    setTaskPagination({ current: 1, pageSize: size });
                  },
                }}
              />
            )}
          </Card>
        </Col>
      </Row>
    </div>
  );

  // 降噪测评 Tab 暂时隐藏，代码保留于 git 历史中

  const getRestorationAlgoLabel = (algoName: string) => {
    const found = restorationAlgorithms.find(a => a.name === algoName);
    return found ? found.display_name : algoName;
  };

  const renderRestorationTab = () => (
    <div>
      <Row gutter={[24, 24]}>
        <Col span={24}>
          <Card
            className="home-card"
            title={<Space><ToolOutlined /><span>音频修复</span></Space>}
            variant="borderless"
            style={{ borderRadius: 12 }}
          >
            <Text type="secondary" style={{ fontSize: 13, marginBottom: 16, display: 'block' }}>
              💡 上传带噪音频，选择降噪算法，即可试听对比处理前后的效果
            </Text>

            <div style={{ marginBottom: 16 }}>
              <Title level={5}>选择修复算法</Title>
              <Select
                value={selectedRestorationAlgorithm || undefined}
                onChange={(val) => setSelectedRestorationAlgorithm(val)}
                placeholder="请选择修复算法"
                style={{ width: '100%', maxWidth: 500 }}
                size="large"
                showSearch
                optionFilterProp="label"
                options={[
                  {
                    label: '🎙️ 语音降噪（深度学习）',
                    options: restorationAlgorithms
                      .filter(a => a.name.includes('clearvoice') && !a.name.includes('_sr_'))
                      .map(a => ({ label: a.display_name, value: a.name })),
                  },
                  {
                    label: '🔊 超分辨率',
                    options: restorationAlgorithms
                      .filter(a => a.name.includes('_sr_') || a.name === 'super_resolution')
                      .map(a => ({ label: a.display_name, value: a.name })),
                  },
                  {
                    label: '⚡ 传统方法（无需模型）',
                    options: restorationAlgorithms
                      .filter(a => a.type === '传统方法')
                      .map(a => ({ label: a.display_name, value: a.name })),
                  },
                  {
                    label: '🤖 其他深度学习',
                    options: restorationAlgorithms
                      .filter(a => a.type === '深度学习'
                        && !a.name.includes('clearvoice')
                        && a.name !== 'super_resolution'
                        && a.name !== 'dereverberation')
                      .map(a => ({ label: a.display_name, value: a.name })),
                  },
                  {
                    label: '🏠 其他修复',
                    options: restorationAlgorithms
                      .filter(a => a.name === 'dereverberation')
                      .map(a => ({ label: a.display_name, value: a.name })),
                  },
                ].filter(g => g.options.length > 0)}
              />
              {/* 选中算法的简介 */}
              {selectedRestorationAlgorithm && (() => {
                const algo = restorationAlgorithms.find(a => a.name === selectedRestorationAlgorithm);
                if (!algo) return null;
                return (
                  <div style={{ marginTop: 8, padding: '8px 12px', background: '#f0f4ff', borderRadius: 6, border: '1px solid #dbeafe' }}>
                    <Text type="secondary" style={{ fontSize: 12 }}>{algo.description}</Text>
                  </div>
                );
              })()}
            </div>

            <div style={{ marginBottom: 16 }}>
              <Title level={5}>上传音频文件</Title>
              <Upload.Dragger
                multiple
                accept=".wav,.mp3,.flac"
                beforeUpload={(file) => {
                  setRestorationFiles(prev => {
                    if (prev.find(f => f.name === file.name && f.size === file.size)) return prev;
                    return [...prev, file];
                  });
                  return false;
                }}
                onRemove={(file) => {
                  setRestorationFiles(prev => prev.filter(f => f.name !== file.name || f.size !== file.size));
                }}
                fileList={restorationFiles.map((f, i) => ({
                  uid: `-${i}`,
                  name: f.name,
                  status: 'done' as const,
                  size: f.size,
                } as any))}
                style={{ marginBottom: 0 }}
              >
                <p className="ant-upload-drag-icon">
                  <ToolOutlined style={{ color: '#667eea', fontSize: 36 }} />
                </p>
                <p className="ant-upload-text" style={{ color: '#1e293b' }}>
                  点击或拖拽音频文件到此处（支持批量）
                </p>
                <p className="ant-upload-hint">
                  支持 .wav / .mp3 / .flac 格式，可同时选择多个文件
                </p>
              </Upload.Dragger>
              {restorationFiles.length > 0 && (
                <Text type="secondary" style={{ fontSize: 12, marginTop: 4, display: 'block' }}>
                  已选择 {restorationFiles.length} 个文件，共 {(restorationFiles.reduce((s, f) => s + f.size, 0) / 1024).toFixed(1)} KB
                </Text>
              )}
            </div>

            <Button
              type="primary"
              onClick={handleRestorationSubmit}
              loading={restorationLoading}
              disabled={restorationFiles.length === 0 || !selectedRestorationAlgorithm}
              block
              size="large"
              icon={<ToolOutlined />}
              style={{
                background: restorationFiles.length === 0 || !selectedRestorationAlgorithm
                  ? 'linear-gradient(135deg, #d9d9d9 0%, #bfbfbf 100%)'
                  : 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
                border: 'none',
                borderRadius: 10,
                height: 48,
              }}
            >
              {restorationLoading ? '处理中...' : '开始修复'}
            </Button>
          </Card>
        </Col>

        <Col span={24}>
          <Card
            className="home-card"
            title={<Space><FileExcelOutlined /><span>音频修复任务列表</span></Space>}
            variant="borderless"
            style={{ borderRadius: 12 }}
            extra={<Button onClick={loadRestorationTasks} icon={<SyncOutlined />}>刷新</Button>}
          >
            {restorationTasks.length === 0 ? (
              <Empty image={Empty.PRESENTED_IMAGE_SIMPLE} description="暂无修复任务" />
            ) : (
              <>
                <Table
                  dataSource={[...restorationTasks].reverse()}
                  columns={restorationColumns}
                  rowKey="task_id"
                  size="small"
                  pagination={{ pageSize: 5 }}
                />
                {/* ── 展开的试听对比面板 ── */}
                {expandedRestorationTask && (() => {
                  const task = restorationTasks.find(t => t.task_id === expandedRestorationTask);
                  if (!task || task.status !== 'completed') return null;
                  const sourceUrl = restorationApi.getSourceAudioUrl(task.task_id);
                  const resultUrl = restorationApi.getResultAudioUrl(task.task_id);
                  return (
                    <div style={{
                      marginTop: 16,
                      padding: 20,
                      background: '#f8fafc',
                      borderRadius: 12,
                      border: '1px solid #e2e8f0',
                    }}>
                      <div style={{ marginBottom: 12, display: 'flex', alignItems: 'center', gap: 16, flexWrap: 'wrap' }}>
                        <Text strong style={{ color: '#1e293b', fontSize: 14 }}>
                          📄 {task.filename}
                        </Text>
                        <Text style={{ color: '#64748b', fontSize: 13 }}>
                          算法: {getRestorationAlgoLabel(task.algorithm)}
                        </Text>
                        {task.processing_time && (
                          <Text style={{ color: '#64748b', fontSize: 13 }}>
                            耗时: {task.processing_time.toFixed(2)}s
                          </Text>
                        )}
                      </div>
                      <AudioComparison sourceUrl={sourceUrl} resultUrl={resultUrl} />
                    </div>
                  );
                })()}
              </>
            )}
          </Card>
        </Col>
      </Row>
    </div>
  );

  const tabItems = [
    {
      key: 'mos',
      label: <Space><SoundOutlined />MOS评分</Space>,
      children: renderMosTab(),
    },
    {
      key: 'restoration',
      label: <Space><ToolOutlined />音频修复</Space>,
      children: renderRestorationTab(),
    },
  ];

  return (
    <Layout className="home-container" style={{ minHeight: '100vh' }}>
      <div className="home-background">
        <div className="bg-gradient-circle circle-1"></div>
        <div className="bg-gradient-circle circle-2"></div>
        <div className="bg-gradient-circle circle-3"></div>
        <div className="bg-gradient-circle circle-4"></div>
        <div className="bg-gradient-circle circle-5"></div>

        <div className="floating-elements">
          <div className="float-item float-1"></div>
          <div className="float-item float-2"></div>
          <div className="float-item float-3"></div>
          <div className="float-item float-4"></div>
          <div className="float-item float-5"></div>
        </div>

        <div className="grid-pattern"></div>

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
          <Tabs
            activeKey={activeTab}
            onChange={setActiveTab}
            items={tabItems}
            size="large"
            style={{
              background: 'transparent',
            }}
          />
        </div>
      </Content>

      <Footer style={{ textAlign: 'center', background: 'transparent' }}>
        <Text type="secondary">
          AudioMOS ©{new Date().getFullYear()} | 音频质量评分系统
        </Text>
      </Footer>

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
                onShowSizeChange: (_current, size) => {
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