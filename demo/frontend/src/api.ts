export const API_BASE = import.meta.env.VITE_API_BASE ?? 'http://localhost:8000';

export type CreateSessionResponse = {
  session_id: string;
  video_url: string;
  orig_width: number;
  orig_height: number;
  num_frames: number;
  fps: number;
  query_frame_index: number;
  has_gt: boolean;
};

export type GetQueryPointsResponse = {
  session_id: string;
  query_frame_index: number;
  gt_pos?: Array<{ x: number; y: number }>;
  user_pos?: Array<{ x: number; y: number }>;
  user_neg?: Array<{ x: number; y: number }>;
};

async function apiError(response: Response): Promise<Error> {
  const body = await response.json().catch(() => null);
  const detail = body?.detail;
  return new Error(typeof detail === 'string' ? detail : `Request failed (${response.status})`);
}

export async function createSession(params: { sequenceFile: File; gtFile?: File | null }): Promise<CreateSessionResponse> {
  const fd = new FormData();
  fd.append('sequence', params.sequenceFile);
  if (params.gtFile) fd.append('gt', params.gtFile);

  const res = await fetch(`${API_BASE}/api/session/create`, {
    method: 'POST',
    body: fd,
    cache: 'no-store',
  });
  if (!res.ok) throw await apiError(res);
  return res.json();
}

export async function getQueryPoints(params: { sessionId: string }): Promise<GetQueryPointsResponse> {
  const res = await fetch(`${API_BASE}/api/session/${params.sessionId}/query_points`, { cache: 'no-store' });
  if (!res.ok) throw await apiError(res);
  return res.json();
}

export async function addQueryPoint(params: {
  sessionId: string;
  x: number;
  y: number;
  label: 1 | 0;
}): Promise<void> {
  const res = await fetch(`${API_BASE}/api/session/${params.sessionId}/query_points/add`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ x: params.x, y: params.y, label: params.label }),
    cache: 'no-store',
  });
  if (!res.ok) throw await apiError(res);
}

export async function resetUserQueryPoints(params: { sessionId: string }): Promise<void> {
  const res = await fetch(`${API_BASE}/api/session/${params.sessionId}/query_points/reset`, {
    method: 'POST',
    cache: 'no-store',
  });
  if (!res.ok) throw await apiError(res);
}

export async function trackObjects(params: {
  sessionId: string;
  useCpd: boolean;
  foregroundClass?: number;
}): Promise<{ ok: boolean; job_id: string }> {
  const res = await fetch(`${API_BASE}/api/session/${params.sessionId}/track`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      use_cpd: params.useCpd,
      foreground_class: params.foregroundClass ?? 1,
    }),
    cache: 'no-store',
  });
  if (!res.ok) throw await apiError(res);
  return res.json();
}

export type TrackStatus =
  | { status: 'idle' }
  | {
      job_id: string;
      status: 'queued' | 'running' | 'succeeded' | 'failed' | 'canceled';
      progress: number;
      message?: string;
      outputs?: Record<string, string>;
    };

export async function getTrackStatus(params: { sessionId: string }): Promise<TrackStatus> {
  const res = await fetch(`${API_BASE}/api/session/${params.sessionId}/track/status`, { cache: 'no-store' });
  if (!res.ok) throw await apiError(res);
  return res.json();
}

export type FrameEditOp = 'add' | 'remove' | 'apply' | 'noop';

export type RLE = { size: number[]; counts: string; order?: string };

export type EditFrameResponse = {
  ok: boolean;
  message?: string;
  applied?: boolean;
  rle?: RLE | null;
};

export async function editFrame(params: {
  sessionId: string;
  frameIndex: number;
  op: FrameEditOp;
  x: number;
  y: number;
  useCpd: boolean;
  maxRemoveDist?: number | null;
  apply?: boolean;
}): Promise<EditFrameResponse> {
  const res = await fetch(`${API_BASE}/api/session/${params.sessionId}/frames/${params.frameIndex}/edit`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      op: params.op,
      x: params.x,
      y: params.y,
      use_cpd: params.useCpd,
      max_remove_dist: params.maxRemoveDist ?? null,
      apply: params.apply ?? false,
    }),
    cache: 'no-store',
  });
  if (!res.ok) throw await apiError(res);
  return res.json();
}

export async function downloadSessionZip(params: { sessionId: string }): Promise<Blob> {
  const res = await fetch(`${API_BASE}/api/session/${params.sessionId}/download.zip`, {
    method: 'GET',
    cache: 'no-store',
  });
  if (!res.ok) throw await apiError(res);
  return res.blob();
}



export async function saveFrameEdits(params: {sessionId: string; frameIndex: number; adds: Array<{x: number; y: number}>; removes: Array<{x: number; y: number}>}): Promise<void> {
  const response = await fetch(`${API_BASE}/api/session/${params.sessionId}/frames/${params.frameIndex}/edits`, {
    method: 'POST', headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({adds: params.adds, removes: params.removes}),
  });
  if (!response.ok) throw await apiError(response);
}
