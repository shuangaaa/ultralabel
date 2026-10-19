import React, { useEffect, useMemo, useRef, useState } from 'react';
import {
  API_BASE,
  addQueryPoint,
  createSession,
  downloadSessionZip,
  getQueryPoints,
  getTrackStatus,
  resetUserQueryPoints,
  trackObjects,
  editFrame,
  saveFrameEdits,
  type RLE,
} from './api';

type Mode = 'query' | 'refine';
type UiPoint = { x: number; y: number };
type PendingAction = null | 'add' | 'remove';

function clamp01(v: number) {
  return Math.max(0, Math.min(1, v));
}

function rleToMaskImageData(rle: RLE, color = { r: 255, g: 0, b: 0 }, alpha = 90): ImageData {
  const [h, w] = rle.size;
  const countsStr = rle.counts;

  const counts: number[] = [];
  let p = 0;
  let m = 0;
  while (p < countsStr.length) {
    let x = 0;
    let k = 0;
    let more = 1;
    while (more) {
      const c = countsStr.charCodeAt(p) - 48;
      p++;
      x |= (c & 0x1f) << (5 * k);
      more = c & 0x20;
      k++;
      if (!more && (c & 0x10)) {
        x |= -1 << (5 * k);
      }
    }
    if (m > 2) x += counts[m - 2];
    counts.push(x);
    m++;
  }

  const img = new ImageData(w, h);
  const data = img.data;

  let idx = 0;
  let val = 0;
  for (let i = 0; i < counts.length; i++) {
    const run = counts[i];
    for (let j = 0; j < run; j++) {
      const lin = idx;
      const y = lin % h;
      const x = (lin - y) / h;

      const off = (y * w + x) * 4;
      if (val === 1) {
        data[off + 0] = color.r;
        data[off + 1] = color.g;
        data[off + 2] = color.b;
        data[off + 3] = alpha;
      } else {
        data[off + 3] = 0;
      }
      idx++;
    }
    val = 1 - val;
  }
  return img;
}

function removeNearestLocal(pts: UiPoint[], x: number, y: number, maxDistNorm = 0.03) {
  if (pts.length === 0) return pts;
  let best = 0;
  let bestD2 = Infinity;
  for (let i = 0; i < pts.length; i++) {
    const dx = pts[i].x - x;
    const dy = pts[i].y - y;
    const d2 = dx * dx + dy * dy;
    if (d2 < bestD2) {
      bestD2 = d2;
      best = i;
    }
  }
  const d = Math.sqrt(bestD2);
  if (d > maxDistNorm) return pts;
  return pts.filter((_, i) => i !== best);
}

function toneForStatus(st: string): { bg: string; fg: string; border: string } {
  if (st === 'succeeded') return { bg: '#ECFDF5', fg: '#065F46', border: '#A7F3D0' };
  if (st === 'failed' || st === 'canceled') return { bg: '#FEF2F2', fg: '#991B1B', border: '#FECACA' };
  if (st === 'queued' || st === 'running') return { bg: '#EFF6FF', fg: '#1D4ED8', border: '#BFDBFE' };
  return { bg: '#F1F5F9', fg: '#334155', border: '#E2E8F0' };
}

export default function App() {
  const wrapRef = useRef<HTMLDivElement | null>(null);
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  const [sequenceFile, setSequenceFile] = useState<File | null>(null);
  const [gtFile, setGtFile] = useState<File | null>(null);

  const [sessionId, setSessionId] = useState<string | null>(null);
  const [hasGt, setHasGt] = useState<boolean>(false);
  const [videoUrl, setVideoUrl] = useState<string | null>(null);

  const [fps, setFps] = useState<number>(5);
  const [numFrames, setNumFrames] = useState<number>(0);
  const [queryFrameIndex, setQueryFrameIndex] = useState<number>(0);
  const [frameIndex, setFrameIndex] = useState<number>(0);

  const [mode, setMode] = useState<Mode>('query');
  const [canvasVersion, setCanvasVersion] = useState(0);
  const [pendingAction, setPendingAction] = useState<PendingAction>(null);

  const [useCpd, setUseCpd] = useState<boolean>(true);

  const [gtQueryPoints, setGtQueryPoints] = useState<UiPoint[]>([]);
  const [userQueryPoints, setUserQueryPoints] = useState<UiPoint[]>([]);

  const [refineAdds, setRefineAdds] = useState<Record<number, UiPoint[]>>({});
  const [refineRemoves, setRefineRemoves] = useState<Record<number, UiPoint[]>>({});

  const [trackStatus, setTrackStatus] = useState<string>('idle');
  const [trackProgress, setTrackProgress] = useState<number>(0);

  const [busy, setBusy] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  const [maskSource, setMaskSource] = useState<string | null>(null);
  const pollTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const sessionToken = useRef(0);

  const [maskRle, setMaskRle] = useState<RLE | null>(null);
  const [maskEnabled, setMaskEnabled] = useState<boolean>(true);

  const MAX_REMOVE_DIST_PX = 25;

  const activeVideoUrl = useMemo(() => {
    const u = videoUrl;
    if (!u) return null;
    if (u.startsWith('http')) return u;
    const bust = `t=${Date.now()}`;
    return `${API_BASE}${u}${u.includes('?') ? '&' : '?'}${bust}`;
  }, [videoUrl]);

  const refinePendingCount = (refineAdds[frameIndex]?.length ?? 0) + (refineRemoves[frameIndex]?.length ?? 0);

  const canEditCanvas =
    Boolean(sessionId && pendingAction) &&
    ((mode === 'query' && frameIndex === queryFrameIndex) || mode === 'refine');

  const statusTone = toneForStatus(trackStatus);

  const styles: Record<string, React.CSSProperties> = {
    page: {
      minHeight: '100vh',
      background: 'linear-gradient(180deg, #F8FAFC 0%, #FFFFFF 60%)',
      color: '#0F172A',
      fontFamily:
        'ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, "Apple Color Emoji","Segoe UI Emoji"',
    },
    container: {
      maxWidth: 1200,
      margin: '0 auto',
      padding: 20,
    },
    headerRow: {
      display: 'flex',
      justifyContent: 'space-between',
      alignItems: 'flex-end',
      gap: 16,
      flexWrap: 'wrap',
    },
    title: { margin: 0, fontSize: 22, letterSpacing: -0.2 },
    subtitle: { marginTop: 6, color: '#475569', fontSize: 13, lineHeight: 1.6 },
    badgeRow: { display: 'flex', gap: 8, flexWrap: 'wrap', alignItems: 'center' },
    badge: {
      display: 'inline-flex',
      alignItems: 'center',
      gap: 8,
      padding: '6px 10px',
      borderRadius: 999,
      border: '1px solid #E2E8F0',
      background: '#F8FAFC',
      fontSize: 12,
      color: '#334155',
    },
    badgeStrong: { fontWeight: 700, color: '#0F172A' },
    badgeStatus: {
      display: 'inline-flex',
      alignItems: 'center',
      gap: 8,
      padding: '6px 10px',
      borderRadius: 999,
      border: `1px solid ${statusTone.border}`,
      background: statusTone.bg,
      fontSize: 12,
      color: statusTone.fg,
      fontWeight: 600,
    },

    grid: {
      marginTop: 16,
      display: 'grid',
      gridTemplateColumns: '1.15fr 1.6fr',
      gap: 14,
    },
    card: {
      background: '#FFFFFF',
      border: '1px solid #E2E8F0',
      borderRadius: 16,
      boxShadow: '0 8px 24px rgba(15, 23, 42, 0.06)',
      overflow: 'hidden',
    },
    cardHeader: {
      padding: '12px 14px',
      borderBottom: '1px solid #F1F5F9',
      display: 'flex',
      justifyContent: 'space-between',
      alignItems: 'center',
      gap: 10,
      background: 'linear-gradient(180deg, #FFFFFF 0%, #FAFBFF 100%)',
    },
    cardTitle: { fontSize: 13, color: '#334155', fontWeight: 800, letterSpacing: 0.2 },
    cardBody: { padding: 14 },

    formRow: { display: 'flex', gap: 10, flexWrap: 'wrap', alignItems: 'center' },
    field: { display: 'flex', flexDirection: 'column', gap: 6, minWidth: 260, flex: '1 1 260px' },
    label: { fontSize: 12, color: '#475569', fontWeight: 700 },
    file: {
      width: '100%',
      padding: 10,
      borderRadius: 12,
      border: '1px solid #E2E8F0',
      background: '#FFFFFF',
      fontSize: 13,
    },

    btnRow: { display: 'flex', gap: 10, flexWrap: 'wrap', alignItems: 'center' },
    btn: {
      padding: '10px 12px',
      borderRadius: 12,
      border: '1px solid #E2E8F0',
      background: '#FFFFFF',
      color: '#0F172A',
      fontWeight: 700,
      fontSize: 13,
      cursor: 'pointer',
      transition: 'transform 0.05s ease, box-shadow 0.15s ease, background 0.15s ease',
      boxShadow: '0 1px 0 rgba(15, 23, 42, 0.02)',
    },
    btnPrimary: {
      border: '1px solid #0284C7',
      background: 'linear-gradient(180deg, #0EA5E9 0%, #0284C7 100%)',
      color: '#FFFFFF',
      boxShadow: '0 10px 20px rgba(2, 132, 199, 0.18)',
    },
    btnDanger: {
      border: '1px solid #F1F5F9',
      background: '#0F172A',
      color: '#FFFFFF',
    },
    btnActive: {
      outline: '2px solid rgba(14, 165, 233, 0.35)',
      outlineOffset: 2,
    },
    btnDisabled: {
      opacity: 0.5,
      cursor: 'not-allowed',
    },

    split: { display: 'flex', gap: 10, flexWrap: 'wrap', alignItems: 'center', justifyContent: 'space-between' },
    hint: { fontSize: 12, color: '#64748B', lineHeight: 1.6 },

    progressWrap: {
      width: '100%',
      height: 8,
      background: '#F1F5F9',
      borderRadius: 999,
      overflow: 'hidden',
      border: '1px solid #E2E8F0',
    },
    progressBar: {
      height: '100%',
      width: `${Math.round(Math.max(0, Math.min(1, trackProgress)) * 100)}%`,
      background: 'linear-gradient(90deg, #0EA5E9 0%, #2563EB 100%)',
    },

    check: { display: 'inline-flex', gap: 8, alignItems: 'center', fontSize: 13, color: '#0F172A' },

    error: {
      marginTop: 14,
      padding: 12,
      borderRadius: 14,
      background: '#FEF2F2',
      border: '1px solid #FECACA',
      color: '#991B1B',
      whiteSpace: 'pre-wrap',
      fontSize: 13,
      lineHeight: 1.55,
    },

    viewerCardBody: { padding: 0 },
    viewerWrap: {
      position: 'relative',
      width: '100%',
      background: '#0B1220',
      overflow: 'hidden',
    },
    viewerTopBar: {
      position: 'absolute',
      left: 12,
      top: 12,
      zIndex: 2,
      display: 'flex',
      gap: 8,
      flexWrap: 'wrap',
      alignItems: 'center',
      padding: '8px 10px',
      borderRadius: 14,
      background: 'rgba(15, 23, 42, 0.60)',
      border: '1px solid rgba(148, 163, 184, 0.25)',
      color: '#E2E8F0',
      backdropFilter: 'blur(6px)',
      fontSize: 12,
    },
    viewerTopDot: {
      width: 8,
      height: 8,
      borderRadius: 999,
      background: trackStatus === 'succeeded' ? '#22C55E' : trackStatus === 'failed' ? '#EF4444' : '#60A5FA',
      boxShadow: '0 0 0 3px rgba(255,255,255,0.06)',
    },

    video: { width: '100%', height: 'auto', display: 'block' },
    empty: { color: '#E2E8F0', padding: 26 },

    legend: { marginTop: 12, color: '#334155', fontSize: 13, lineHeight: 1.65 },
    legendList: { marginTop: 8, paddingLeft: 18, color: '#475569' },
  };

  function syncCanvasSize() {
    const video = videoRef.current;
    const canvas = canvasRef.current;
    if (!video || !canvas) return;
    const rect = video.getBoundingClientRect();
    const w = Math.max(1, Math.floor(rect.width));
    const h = Math.max(1, Math.floor(rect.height));
    if (canvas.width !== w || canvas.height !== h) {
      canvas.width = w;
      canvas.height = h;
      setCanvasVersion(previous => previous + 1);
    }
  }

  useEffect(() => {
    const onResize = () => syncCanvasSize();
    window.addEventListener('resize', onResize);
    return () => window.removeEventListener('resize', onResize);
  }, []);

  useEffect(() => {
    const el = wrapRef.current;
    if (!el) return;
    const ro = new ResizeObserver(() => syncCanvasSize());
    ro.observe(el);
    syncCanvasSize();
    return () => ro.disconnect();
  }, [activeVideoUrl]);

  useEffect(() => {
    const onKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape') setPendingAction(null);
    };
    window.addEventListener('keydown', onKeyDown);
    return () => window.removeEventListener('keydown', onKeyDown);
  }, []);

  function updateFrameIndexFromVideo() {
    const v = videoRef.current;
    if (!v) return;
    const idx = Math.floor(v.currentTime * fps);
    setFrameIndex(Math.max(0, Math.min(numFrames > 0 ? numFrames - 1 : idx, idx)));
  }

  function getClickNormalized(e: React.MouseEvent<HTMLCanvasElement>) {
    const canvas = canvasRef.current;
    if (!canvas) return null;
    const rect = canvas.getBoundingClientRect();
    const nx = clamp01((e.clientX - rect.left) / rect.width);
    const ny = clamp01((e.clientY - rect.top) / rect.height);
    return { nx, ny };
  }

  async function refreshQueryPointsFromServer(sid: string) {
    const qp = await getQueryPoints({ sessionId: sid });
    setGtQueryPoints((qp.gt_pos ?? []).map(p => ({ x: p.x, y: p.y })));
    setUserQueryPoints((qp.user_pos ?? []).map(p => ({ x: p.x, y: p.y })));
  }

  async function fetchMaskRleForFrame(sid: string, fi: number) {
    try {
      const bust = `t=${Date.now()}`;
      const res = await fetch(`${API_BASE}/api/session/${sid}/frames/${fi}/mask-rle?${bust}`, { cache: 'no-store' });
      if (!res.ok) throw new Error(await res.text());
      const j = await res.json();
      setMaskRle(j.rle as RLE);
      setMaskSource(j.source ?? null);
    } catch {
      setMaskRle(null);
      setMaskSource(null);
    }
  }

  async function onCreateSession() {
    if (!sequenceFile) {
      setError('请先选择视频或 NIfTI 序列文件');
      return;
    }
    try {
      setBusy(true);
      setError(null);

      sessionToken.current += 1;
      if (pollTimer.current) clearTimeout(pollTimer.current);
      setNotice(null);
      setMaskSource(null);
      setSessionId(null);
      setVideoUrl(null);
      setTrackStatus('idle');
      setTrackProgress(0);

      setGtQueryPoints([]);
      setUserQueryPoints([]);
      setRefineAdds({});
      setRefineRemoves({});
      setPendingAction(null);
      setMaskRle(null);

      const s = await createSession({ sequenceFile, gtFile });
      setSessionId(s.session_id);
      setVideoUrl(s.video_url);
      setHasGt(s.has_gt);
      setUseCpd(s.has_gt);
      setFps(s.fps);
      setNumFrames(s.num_frames);
      setQueryFrameIndex(s.query_frame_index);
      setFrameIndex(s.query_frame_index);
      setMode('query');

      await refreshQueryPointsFromServer(s.session_id);
      await fetchMaskRleForFrame(s.session_id, s.query_frame_index);
    } catch (e: any) {
      setError(e?.message ?? String(e));
    } finally {
      setBusy(false);
    }
  }

  async function onClickCanvas(e: React.MouseEvent<HTMLCanvasElement>) {
    if (!sessionId) return;
    if (!pendingAction) return;

    videoRef.current?.pause();
    const pos = getClickNormalized(e);
    if (!pos) return;
    const { nx, ny } = pos;

    if (mode === 'query') {
      if (frameIndex !== queryFrameIndex) {
        setError(`A模式编辑只允许在 query_frame=${queryFrameIndex}。请拖动视频到该帧。`);
        return;
      }
      setError(null);

      if (pendingAction === 'add') {
        const next = [...userQueryPoints, { x: nx, y: ny }];
        setUserQueryPoints(next);
        try {
          await addQueryPoint({ sessionId, x: nx, y: ny, label: 1 });
        } catch (e: any) {
          setError(e?.message ?? String(e));
        }
      } else {
        const next = removeNearestLocal(userQueryPoints, nx, ny);
        setUserQueryPoints(next);
        try {
          await resetUserQueryPoints({ sessionId });
          for (const p of next) {
            await addQueryPoint({ sessionId, x: p.x, y: p.y, label: 1 });
          }
        } catch (e: any) {
          setError(e?.message ?? String(e));
        }
      }
      return;
    }

    if (mode === 'refine') {
      setError(null);
      if (pendingAction === 'add') {
        const cur = refineAdds[frameIndex] ?? [];
        setRefineAdds(prev => ({ ...prev, [frameIndex]: [...cur, { x: nx, y: ny }] }));
      } else {
        const cur = refineRemoves[frameIndex] ?? [];
        setRefineRemoves(prev => ({ ...prev, [frameIndex]: [...cur, { x: nx, y: ny }] }));
      }
      return;
    }
  }

  async function onResetUserQueryPoints() {
    if (!sessionId) return;
    try {
      setBusy(true);
      setError(null);
      setPendingAction(null);
      await resetUserQueryPoints({ sessionId });
      await refreshQueryPointsFromServer(sessionId);
    } catch (e: any) {
      setError(e?.message ?? String(e));
    } finally {
      setBusy(false);
    }
  }

  async function onTrack() {
    if (!sessionId) return;
    try {
      setBusy(true);
      setError(null);
      setPendingAction(null);

      if (!hasGt && userQueryPoints.length === 0) {
        throw new Error('未上传GT且查询点为空：请在 query frame 先添加至少一个查询点。');
      }

      await trackObjects({
        sessionId,
        useCpd: useCpd && hasGt,
        foregroundClass: 1,
      });

      setTrackStatus('queued');
      setTrackProgress(0);

      const token = sessionToken.current;
      const poll = async () => {
        if (token !== sessionToken.current) return;
        try {
        const st = await getTrackStatus({ sessionId });
        if ((st as any).status === 'idle') return;

        setTrackStatus((st as any).status);
        setTrackProgress((st as any).progress ?? 0);

        if ((st as any).status === 'succeeded') {
          setMode('refine');
          await fetchMaskRleForFrame(sessionId, frameIndex);
          return;
        }
        if ((st as any).status === 'failed' || (st as any).status === 'canceled') {
          setError((st as any).message ?? 'track failed');
          return;
        }
        pollTimer.current = setTimeout(poll, 800);
        } catch (error: any) {
          if (token === sessionToken.current) {setTrackStatus('failed'); setError(error?.message ?? String(error));}
        }
      };
      pollTimer.current = setTimeout(poll, 300);
    } catch (e: any) {
      setError(e?.message ?? String(e));
    } finally {
      setBusy(false);
    }
  }

  useEffect(() => () => {
    sessionToken.current += 1;
    if (pollTimer.current) clearTimeout(pollTimer.current);
  }, []);

  async function onSaveFrameEdits() {
    if (!sessionId) return;
    try {
      setError(null);
      await saveFrameEdits({sessionId, frameIndex, adds: refineAdds[frameIndex] ?? [], removes: refineRemoves[frameIndex] ?? []});
      setNotice('当前帧的修正点已保存。');
    } catch (error: any) {setError(error?.message ?? String(error));}
  }

  async function onApplyRefineFrame() {
    if (!sessionId) return;
    try {
      setBusy(true);
      setError(null);
      await saveFrameEdits({sessionId, frameIndex, adds: refineAdds[frameIndex] ?? [], removes: refineRemoves[frameIndex] ?? []});
      const response = await editFrame({sessionId, frameIndex, op: 'apply', x: 0, y: 0, useCpd: useCpd && hasGt, apply: true});
      setRefineAdds(previous => ({...previous, [frameIndex]: []}));
      setRefineRemoves(previous => ({...previous, [frameIndex]: []}));
      if (response.rle) {setMaskRle(response.rle); setMaskSource('prediction');}
      else await fetchMaskRleForFrame(sessionId, frameIndex);
    } catch (error: any) {setError(error?.message ?? String(error));}
    finally {setBusy(false);}
  }

  async function onDownloadZip() {
    if (!sessionId) return;
    try {
      setError(null);
      const blob = await downloadSessionZip({ sessionId });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `session_${sessionId}.zip`;
      document.body.appendChild(a);
      a.click();
      a.remove();
      URL.revokeObjectURL(url);
    } catch (e: any) {
      setError(e?.message ?? String(e));
    }
  }

  useEffect(() => {
    if (!sessionId) return;
    fetchMaskRleForFrame(sessionId, frameIndex);
  }, [sessionId, frameIndex, trackStatus]);

  useEffect(() => {
    syncCanvasSize();
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    ctx.clearRect(0, 0, canvas.width, canvas.height);

    if (maskEnabled && maskRle) {
      const img = rleToMaskImageData(maskRle, { r: 255, g: 0, b: 0 }, 90);
      const tmp = document.createElement('canvas');
      tmp.width = img.width;
      tmp.height = img.height;
      const tctx = tmp.getContext('2d');
      if (tctx) {
        tctx.putImageData(img, 0, 0);
        ctx.drawImage(tmp, 0, 0, canvas.width, canvas.height);
      }
    }

    const drawPts = (pts: UiPoint[], color: string, radius = 6) => {
      for (const p of pts) {
        ctx.beginPath();
        ctx.arc(p.x * canvas.width, p.y * canvas.height, radius, 0, Math.PI * 2);
        ctx.fillStyle = color;
        ctx.fill();
        ctx.lineWidth = 2;
        ctx.strokeStyle = 'rgba(0,0,0,0.6)';
        ctx.stroke();
      }
    };

    if (frameIndex === queryFrameIndex) {
      drawPts(gtQueryPoints, 'rgba(160,90,255,0.95)', 5);
      drawPts(userQueryPoints, 'rgba(0,220,120,0.95)', 6);
    }

    if (mode === 'refine') {
      const adds = refineAdds[frameIndex] ?? [];
      const rms = refineRemoves[frameIndex] ?? [];
      drawPts(adds, 'rgba(0,160,255,0.9)', 6);
      drawPts(rms, 'rgba(255,160,0,0.9)', 6);
    }

    if (pendingAction) {
      ctx.save();
      ctx.fillStyle = 'rgba(15,23,42,0.55)';
      ctx.fillRect(0, 0, canvas.width, 36);
      ctx.fillStyle = 'white';
      ctx.font = '14px system-ui';
      ctx.fillText(
        pendingAction === 'add' ? '持续加点：点击画面（Esc取消）' : '持续减点：点击画面（Esc取消）',
        12,
        23
      );
      ctx.restore();
    }
  }, [
    mode,
    gtQueryPoints,
    userQueryPoints,
    refineAdds,
    refineRemoves,
    frameIndex,
    queryFrameIndex,
    pendingAction,
    maskRle,
    maskEnabled,
    canvasVersion,
  ]);

  const isMobile = typeof window !== 'undefined' ? window.innerWidth < 980 : false;

  return (
    <div style={styles.page}>
      <div style={styles.container}>
        
        <div style={styles.headerRow}>
          <div>
            <h2 style={styles.title}>UltraLabel Demo</h2>
            <div style={styles.subtitle}>A(仅 query frame 点) → Track → B(单帧 refine + Apply)</div>
          </div>

          <div style={styles.badgeRow}>
            <span style={styles.badge}>
              hasGT: <span style={styles.badgeStrong}>{String(hasGt)}</span>
            </span>
            <span style={styles.badge}>
              frame:{' '}
              <span style={styles.badgeStrong}>
                {frameIndex} / {Math.max(0, numFrames - 1)}
              </span>
            </span>
            <span style={styles.badgeStatus}>
              <span
                style={{
                  width: 8,
                  height: 8,
                  borderRadius: 999,
                  background: trackStatus === 'succeeded' ? '#22C55E' : trackStatus === 'failed' ? '#EF4444' : '#60A5FA',
                }}
              />
              status: {trackStatus} · {Math.round(trackProgress * 100)}%
            </span>
          </div>
        </div>

        <div
          style={{
            ...styles.grid,
            gridTemplateColumns: isMobile ? '1fr' : '1.15fr 1.6fr',
          }}
        >
          
          <div style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>
            <div style={styles.card}>
              <div style={styles.cardHeader}>
                <div style={styles.cardTitle}>1) 数据与会话</div>
                <button
                  style={{
                    ...styles.btn,
                    ...(sessionId ? {} : styles.btnDisabled),
                  }}
                  onClick={onDownloadZip}
                  disabled={!sessionId}
                  title="下载会话标注及已有输出"
                >
                  Download ZIP
                </button>
              </div>
              <div style={styles.cardBody}>
                <div style={styles.formRow}>
                  <div style={styles.field}>
                    <div style={styles.label}>sequence</div>
                    <input
                      style={styles.file}
                      type="file"
                      accept=".nii,.gz,.mp4,.mov,.avi,.webm,video/*,application/gzip"
                      onChange={e => setSequenceFile(e.target.files?.[0] ?? null)}
                    />
                  </div>
                  <div style={styles.field}>
                    <div style={styles.label}>gt (optional)</div>
                    <input
                      style={styles.file}
                      type="file"
                      accept=".nii,.gz,application/gzip"
                      onChange={e => setGtFile(e.target.files?.[0] ?? null)}
                    />
                  </div>
                </div>

                <div style={{ marginTop: 12, ...styles.btnRow }}>
                  <button
                    style={{
                      ...styles.btn,
                      ...styles.btnPrimary,
                      ...(busy || !sequenceFile ? styles.btnDisabled : {}),
                    }}
                    onClick={onCreateSession}
                    disabled={busy || !sequenceFile}
                  >
                    {busy ? '处理中…' : 'Create Session'}
                  </button>

                  <label style={{ ...styles.check, opacity: sessionId ? 1 : 0.6 }}>
                    <input
                      type="checkbox"
                      checked={maskEnabled}
                      onChange={e => setMaskEnabled(e.target.checked)}
                      disabled={!sessionId}
                    />
                    Show mask
                  </label>

                  <div style={{ flex: 1 }} />

                  <div style={{ minWidth: 220 }}>
                    <div style={{ fontSize: 12, color: '#64748B', marginBottom: 6 }}>
                      Track progress: {Math.round(trackProgress * 100)}%
                    </div>
                    <div style={styles.progressWrap}>
                      <div style={styles.progressBar} />
                    </div>
                  </div>
                </div>

                <div style={{ marginTop: 10, ...styles.hint }}>
                  query_frame：<span style={{ fontWeight: 700, color: '#0F172A' }}> {queryFrameIndex}</span>
                </div>
              </div>
            </div>

            <div style={styles.card}>
              <div style={styles.cardHeader}>
                <div style={styles.cardTitle}>2) 模式与编辑</div>
                <button
                  style={{
                    ...styles.btn,
                    ...(busy || !sessionId ? styles.btnDisabled : styles.btnDanger),
                  }}
                  onClick={onTrack}
                  disabled={!sessionId || busy}
                  title="开始 Track（Track 后进入 B 模式 refine）"
                >
                  Track
                </button>
              </div>
              <div style={styles.cardBody}>
                <div style={styles.btnRow}>
                  <button
                    style={{
                      ...styles.btn,
                      ...(mode === 'query' ? styles.btnActive : {}),
                      ...(!sessionId || mode === 'query' ? styles.btnDisabled : {}),
                    }}
                    onClick={() => {
                      setMode('query');
                      setPendingAction(null);
                    }}
                    disabled={!sessionId || mode === 'query'}
                  >
                    A: Edit Query Points
                  </button>

                  <button
                    style={{
                      ...styles.btn,
                      ...(mode === 'refine' ? styles.btnActive : {}),
                      ...(!sessionId || mode === 'refine' ? styles.btnDisabled : {}),
                    }}
                    onClick={() => {
                      setMode('refine');
                      setPendingAction(null);
                    }}
                    disabled={!sessionId || mode === 'refine'}
                  >
                    B: Refine Frame
                  </button>
                </div>

                <div style={{ marginTop: 12, ...styles.btnRow }}>
                  <button
                    style={{
                      ...styles.btn,
                      ...(pendingAction === 'add' ? styles.btnActive : {}),
                      ...(!sessionId ? styles.btnDisabled : {}),
                    }}
                    onClick={() => setPendingAction('add')}
                    disabled={!sessionId}
                    title="进入加点模式（持续添加，Esc 或 Cancel 退出）"
                  >
                    Add point (keep)
                  </button>

                  <button
                    style={{
                      ...styles.btn,
                      ...(pendingAction === 'remove' ? styles.btnActive : {}),
                      ...(!sessionId ? styles.btnDisabled : {}),
                    }}
                    onClick={() => setPendingAction('remove')}
                    disabled={!sessionId}
                    title="进入减点模式（持续点击删除附近点，Esc 或 Cancel 退出）"
                  >
                    Remove point (keep)
                  </button>

                  <button
                    style={{
                      ...styles.btn,
                      ...(!sessionId || !pendingAction ? styles.btnDisabled : {}),
                    }}
                    onClick={() => setPendingAction(null)}
                    disabled={!sessionId || !pendingAction}
                  >
                    Cancel (Esc)
                  </button>
                </div>

                <div style={{ marginTop: 12, ...styles.split }}>
                  <label style={styles.check} title="需要 GT 才可用">
                    <input type="checkbox" checked={useCpd} onChange={e => setUseCpd(e.target.checked)} disabled={!hasGt} />
                    use CPD (need GT)
                  </label>

                  <button
                    style={{
                      ...styles.btn,
                      ...(!sessionId || busy ? styles.btnDisabled : {}),
                    }}
                    onClick={onResetUserQueryPoints}
                    disabled={!sessionId || busy}
                    title="只清除用户新增 query points（绿色点）"
                  >
                    Reset user query points
                  </button>
                </div>

                {mode === 'refine' && (
                  <div style={{ marginTop: 12, ...styles.split }}>
                    <button style={{...styles.btn, ...(!sessionId || busy || refinePendingCount === 0 ? styles.btnDisabled : {})}} disabled={!sessionId || busy || refinePendingCount === 0} onClick={onSaveFrameEdits}>Save edits</button>
                    <button
                      style={{
                        ...styles.btn,
                        ...styles.btnPrimary,
                        ...(!sessionId || trackStatus !== 'succeeded' || busy || refinePendingCount === 0 ? styles.btnDisabled : {}),
                      }}
                      onClick={onApplyRefineFrame}
                      disabled={!sessionId || trackStatus !== 'succeeded' || busy || refinePendingCount === 0}
                      title="Apply：只更新当前帧 mask"
                    >
                      Apply (update this frame mask)
                    </button>

                    <div style={styles.hint}>
                      pending: <b>{refinePendingCount}</b>（add {refineAdds[frameIndex]?.length ?? 0} / remove{' '}
                      {refineRemoves[frameIndex]?.length ?? 0}）
                    </div>
                  </div>
                )}

                <div style={{ marginTop: 10, ...styles.hint }}>
                  {mode === 'query' ? (
                    <>
                      A模式提示：只能在 <b>query frame={queryFrameIndex}</b> 编辑；当前帧为 <b>{frameIndex}</b>。
                    </>
                  ) : (
                    <>B模式提示：记录当前帧的 Add/Remove；Save edits 保存修正点，Apply 更新当前帧标注。</>
                  )}
                </div>
              </div>
            </div>

            {notice && <div role="status" style={{...styles.cardBody, color: '#166534'}}>{notice}</div>}
            {error && <pre role="alert" style={styles.error}>{error}</pre>}
          </div>

          <div style={styles.card}>
            <div style={styles.cardHeader}>
              <div style={styles.cardTitle}>3) 预览</div>
              <div style={{ ...styles.hint, textAlign: 'right' }}>
                canvas：{canEditCanvas ? <b>可编辑</b> : '不可编辑'}（只有选择 Add/Remove 且满足模式条件才可点）
              </div>
            </div>

            <div style={styles.viewerCardBody}>
              <div ref={wrapRef} style={styles.viewerWrap}>
                {activeVideoUrl ? (
                  <>
                    <div style={styles.viewerTopBar}>
                      <span style={styles.viewerTopDot} />
                      <span>
                        Mode: <b>{mode}</b>
                      </span>
                      <span style={{ opacity: 0.7 }}>·</span>
                      <span>
                        Frame: <b>{frameIndex}</b>
                      </span>
                      <span style={{ opacity: 0.7 }}>·</span>
                      <span>
                        Query: <b>{queryFrameIndex}</b>
                      </span>
                    </div>

                    <video
                      ref={videoRef}
                      src={activeVideoUrl}
                      controls
                      preload="auto"
                      playsInline
                      style={styles.video}
                      onLoadedMetadata={syncCanvasSize}
                      onLoadedData={syncCanvasSize}
                      onCanPlay={syncCanvasSize}
                      onTimeUpdate={updateFrameIndexFromVideo}
                    />

                    <canvas
                      ref={canvasRef}
                      onClick={onClickCanvas}
                      style={{
                        position: 'absolute',
                        left: 0,
                        top: 0,
                        width: '100%',
                        height: '100%',
                        cursor: canEditCanvas ? 'crosshair' : 'default',
                        pointerEvents: canEditCanvas ? 'auto' : 'none',
                      }}
                    />
                  </>
                ) : (
                  <div style={styles.empty}>No video yet. Upload and Create Session.</div>
                )}
              </div>

              <div style={{ padding: 14 }}>
                <div style={styles.legend}>
                  <div style={{ fontWeight: 800, color: '#0F172A' }}>颜色说明</div>
                  <ul style={styles.legendList}>
                    <li>紫色点：GT 自动 query points（仅在 query frame 显示）</li>
                    <li>绿色点：用户新增 query points（仅在 query frame 显示；Reset 只清它）</li>
                    <li>蓝色点：B模式待 Apply 的 Add 点（仅当前帧）</li>
                    <li>橙色点：B模式待应用的 Remove 位置（仅当前帧）。</li>
                    <li>红色半透明区域：{maskSource === 'reference' ? '上传的参考标注' : maskSource === 'prediction' ? '计算结果' : '当前帧标注'}。</li>
                  </ul>
                </div>
              </div>
            </div>
          </div>
        </div>

        <div style={{ marginTop: 14, color: '#94A3B8', fontSize: 12, lineHeight: 1.6 }}>
          小提示：如果你在 A 模式且不在 query frame，Canvas 会自动变为不可点击，避免误操作。
        </div>
      </div>
    </div>
  );
}


