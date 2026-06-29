/**
 * 音频试听对比组件（共享）
 * 用于 Home.tsx 和 Restoration.tsx 中展示处理前后的音频对比
 */
import React, { useState, useRef, useEffect } from 'react';
import { Typography } from 'antd';
import { PlayCircleFilled, PauseCircleFilled } from '@ant-design/icons';

const { Text } = Typography;

// ── 配色 ──────────────────────────────────────────────────
const COLORS = {
  source:   { primary: '#f59e0b', light: '#fef3c7', fill: '#fcd34d', text: '#92400e' },
  result:   { primary: '#10b981', light: '#d1fae5', fill: '#6ee7b7', text: '#065f46' },
  arrow:    '#6366f1',
  canvasBg: '#f8fafc',
  cardBg:   '#ffffff',
  border:   '#e2e8f0',
  shadow:   '0 1px 3px rgba(0,0,0,0.06), 0 1px 2px rgba(0,0,0,0.04)',
  timeText: '#94a3b8',
  seekBg:   '#e2e8f0',
};

// ── 波形绘制 Hook ──────────────────────────────────────────
function useWaveform(
  canvasRef: React.RefObject<HTMLCanvasElement | null>,
  audioUrl: string,
  fillColor: string,
  primaryColor: string,
) {
  useEffect(() => {
    if (!audioUrl || !canvasRef.current) return;
    const canvas = canvasRef.current;
    let cancelled = false;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const audioContext = new (window.AudioContext || (window as any).webkitAudioContext)();

    fetch(audioUrl)
      .then((r) => r.arrayBuffer())
      .then((buf) => audioContext.decodeAudioData(buf))
      .then((audioBuffer) => {
        if (cancelled) return;
        const data = audioBuffer.getChannelData(0);
        const w = canvas.width;
        const h = canvas.height;
        const step = Math.ceil(data.length / w);
        const mid = h / 2;

        ctx.clearRect(0, 0, w, h);

        // 背景中线
        ctx.strokeStyle = '#e2e8f0';
        ctx.lineWidth = 1;
        ctx.setLineDash([4, 4]);
        ctx.beginPath();
        ctx.moveTo(0, mid);
        ctx.lineTo(w, mid);
        ctx.stroke();
        ctx.setLineDash([]);

        // 上半波形填充
        ctx.fillStyle = fillColor;
        ctx.beginPath();
        ctx.moveTo(0, mid);
        for (let i = 0; i < w; i++) {
          let max = -1;
          for (let j = 0; j < step; j++) {
            const d = data[Math.floor(i * step + j)];
            if (d !== undefined && d > max) max = d;
          }
          ctx.lineTo(i, mid - max * mid * 0.85);
        }
        ctx.lineTo(w, mid);
        ctx.closePath();
        ctx.fill();

        // 下半波形（较浅）
        ctx.fillStyle = fillColor + '66';
        ctx.beginPath();
        ctx.moveTo(0, mid);
        for (let i = 0; i < w; i++) {
          let min = 1;
          for (let j = 0; j < step; j++) {
            const d = data[Math.floor(i * step + j)];
            if (d !== undefined && d < min) min = d;
          }
          ctx.lineTo(i, mid - min * mid * 0.85);
        }
        ctx.lineTo(w, mid);
        ctx.closePath();
        ctx.fill();

        // 包络线
        ctx.strokeStyle = primaryColor;
        ctx.lineWidth = 1.5;
        ctx.beginPath();
        ctx.moveTo(0, mid);
        for (let i = 0; i < w; i++) {
          let max = -1;
          for (let j = 0; j < step; j++) {
            const d = data[Math.floor(i * step + j)];
            if (d !== undefined && d > max) max = d;
          }
          ctx.lineTo(i, mid - max * mid * 0.85);
        }
        ctx.stroke();
      })
      .catch(() => {});

    return () => {
      cancelled = true;
      audioContext.close().catch(() => {});
    };
  }, [audioUrl]);
}

// ── 单边播放器 ────────────────────────────────────────────
interface AudioSideProps {
  audioUrl: string;
  label: string;
  icon: string;
  colors: typeof COLORS.source;
}

const AudioSide: React.FC<AudioSideProps> = ({ audioUrl, label, icon, colors }) => {
  const audioRef = useRef<HTMLAudioElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [playing, setPlaying] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);

  useWaveform(canvasRef, audioUrl, colors.fill, colors.primary);

  const togglePlay = () => {
    const a = audioRef.current;
    if (!a) return;
    playing ? a.pause() : a.play();
    setPlaying(!playing);
  };

  const handleSeek = (e: React.MouseEvent<HTMLDivElement>) => {
    const a = audioRef.current;
    if (!a || !duration) return;
    const rect = e.currentTarget.getBoundingClientRect();
    a.currentTime = ((e.clientX - rect.left) / rect.width) * duration;
  };

  const fmt = (t: number) => {
    const m = Math.floor(t / 60);
    const s = Math.floor(t % 60);
    return `${m}:${s.toString().padStart(2, '0')}`;
  };

  return (
    <div
      style={{
        background: COLORS.cardBg,
        borderRadius: 12,
        border: `1px solid ${COLORS.border}`,
        boxShadow: COLORS.shadow,
        overflow: 'hidden',
      }}
    >
      {/* 标题栏 */}
      <div
        style={{
          padding: '10px 14px',
          background: colors.light,
          borderBottom: `1px solid ${colors.fill}44`,
          display: 'flex',
          alignItems: 'center',
          gap: 8,
        }}
      >
        <span style={{ fontSize: 16 }}>{icon}</span>
        <Text strong style={{ color: colors.text, fontSize: 13 }}>
          {label}
        </Text>
      </div>

      {/* 波形图 */}
      <div style={{ padding: '8px 12px' }}>
        <canvas
          ref={canvasRef}
          width={520}
          height={72}
          style={{
            width: '100%',
            borderRadius: 6,
            background: COLORS.canvasBg,
            display: 'block',
          }}
        />
      </div>

      {/* 播放控制 */}
      <div
        style={{
          display: 'flex',
          alignItems: 'center',
          gap: 10,
          padding: '6px 14px 12px',
        }}
      >
        {/* 播放按钮 */}
        <div
          onClick={togglePlay}
          style={{
            width: 36,
            height: 36,
            borderRadius: '50%',
            background: `linear-gradient(135deg, ${colors.primary}, ${colors.primary}dd)`,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            cursor: 'pointer',
            boxShadow: `0 2px 8px ${colors.primary}44`,
            transition: 'transform 0.15s, box-shadow 0.15s',
            flexShrink: 0,
          }}
          onMouseEnter={(e) => {
            (e.currentTarget as HTMLDivElement).style.transform = 'scale(1.08)';
          }}
          onMouseLeave={(e) => {
            (e.currentTarget as HTMLDivElement).style.transform = 'scale(1)';
          }}
        >
          {playing ? (
            <PauseCircleFilled style={{ fontSize: 20, color: '#fff' }} />
          ) : (
            <PlayCircleFilled style={{ fontSize: 20, color: '#fff' }} />
          )}
        </div>

        {/* 时间 */}
        <Text style={{ color: COLORS.timeText, fontSize: 12, fontFamily: 'monospace', minWidth: 30 }}>
          {fmt(currentTime)}
        </Text>

        {/* 进度条 */}
        <div
          onClick={handleSeek}
          style={{
            flex: 1,
            height: 6,
            background: COLORS.seekBg,
            borderRadius: 3,
            cursor: 'pointer',
            position: 'relative',
            overflow: 'hidden',
          }}
        >
          <div
            style={{
              height: '100%',
              width: duration ? `${(currentTime / duration) * 100}%` : '0%',
              background: `linear-gradient(90deg, ${colors.primary}, ${colors.primary}cc)`,
              borderRadius: 3,
              transition: 'width 0.05s linear',
            }}
          />
        </div>

        {/* 总时长 */}
        <Text style={{ color: COLORS.timeText, fontSize: 12, fontFamily: 'monospace', minWidth: 30 }}>
          {fmt(duration)}
        </Text>
      </div>

      <audio
        ref={audioRef}
        src={audioUrl}
        preload="metadata"
        onTimeUpdate={() => setCurrentTime(audioRef.current?.currentTime || 0)}
        onLoadedMetadata={() => setDuration(audioRef.current?.duration || 0)}
        onEnded={() => setPlaying(false)}
        onPlay={() => setPlaying(true)}
        onPause={() => setPlaying(false)}
      />
    </div>
  );
};

// ── 对比面板 ───────────────────────────────────────────────
interface AudioComparisonProps {
  sourceUrl: string;
  resultUrl: string;
}

const AudioComparison: React.FC<AudioComparisonProps> = ({ sourceUrl, resultUrl }) => {
  return (
    <div
      style={{
        display: 'grid',
        gridTemplateColumns: '1fr auto 1fr',
        gap: 20,
        alignItems: 'start',
        padding: '8px 0',
      }}
    >
      <AudioSide
        audioUrl={sourceUrl}
        label="原始带噪音频"
        icon="🔊"
        colors={COLORS.source}
      />
      {/* 中间箭头 */}
      <div
        style={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          paddingTop: 60,
        }}
      >
        <div
          style={{
            width: 40,
            height: 40,
            borderRadius: '50%',
            background: 'linear-gradient(135deg, #6366f1, #8b5cf6)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            boxShadow: '0 2px 12px rgba(99,102,241,0.3)',
            fontSize: 18,
            color: '#fff',
          }}
        >
          →
        </div>
      </div>
      <AudioSide
        audioUrl={resultUrl}
        label="修复后音频"
        icon="✨"
        colors={COLORS.result}
      />
    </div>
  );
};

export default AudioComparison;
