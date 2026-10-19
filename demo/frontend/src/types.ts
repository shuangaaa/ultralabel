
export type Label = 0 | 1;

export type NormalizedPoint = {
  x: number;
  y: number;
  label: Label;
};

export type CreateSessionResponse = {
  session_id: string;

  video_url: string;

  infer_width: number;
  infer_height: number;

  orig_width: number;
  orig_height: number;

  num_frames: number;
  fps: number;

  query_frame_index: number;
};

export type InferMode = 'propagate' | 'single';

export type InferRequest = {
  frame_index: number;
  mode: InferMode;
};

export type InferResponse = {
  mask_png_url?: string;

  masks_manifest_url?: string;
};

export type MasksManifest = {
  session_id: string;
  num_frames: number;
  masks: string[];
};
