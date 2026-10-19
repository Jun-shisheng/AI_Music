const BASE = '/api';

export interface ArrangeRequest {
  prompt: string;
  lyrics?: string;
}

export interface StyleSpec {
  genres: string[];
  moods: string[];
  instruments: string[];
  vocals: string[];
  tempo: string;
}

export interface ArrangeResponse {
  title: string;
  language: string;
  lyrics: string;
  style: StyleSpec;
  style_prompt: string;
}

export interface GenerateRequest {
  title?: string;
  lyrics: string;
  style_prompt: string;
  instrumental: boolean;
  generation_type?: 'song' | 'bgm_loop' | 'sfx';
  duration_seconds?: number;
  loopable?: boolean;
  variation_label?: string;
}

export interface JobStatus {
  id: string;
  title: string;
  status: 'queued' | 'running' | 'succeeded' | 'failed';
  lyrics?: string;
  style_prompt?: string;
  instrumental?: boolean;
  generation_type?: 'song' | 'bgm_loop' | 'sfx';
  duration_seconds?: number;
  loopable?: boolean;
  variation_label?: string;
  audio_path?: string;
  audio_url?: string;
  error?: string;
  created_at: string;
  updated_at: string;
}

export interface LibraryItem extends JobStatus {}

function formatErrorDetail(detail: unknown): string | undefined {
  if (typeof detail === 'string') return detail.trim() || undefined;
  if (Array.isArray(detail)) {
    return detail.map(formatErrorDetail).filter(Boolean).join('; ') || undefined;
  }
  if (detail && typeof detail === 'object' && 'msg' in detail && typeof detail.msg === 'string') {
    const location = 'loc' in detail && Array.isArray(detail.loc)
      ? detail.loc.filter((part) => typeof part === 'string' || typeof part === 'number').join('.')
      : '';
    return location ? `${location}: ${detail.msg}` : detail.msg;
  }
  return undefined;
}

async function responseError(res: Response, fallback: string): Promise<Error> {
  const body: unknown = await res.json().catch(() => null);
  const detail = body && typeof body === 'object' && 'detail' in body ? body.detail : undefined;
  return new Error(formatErrorDetail(detail) || res.statusText || fallback);
}

export async function arrange(req: ArrangeRequest): Promise<ArrangeResponse> {
  const res = await fetch(`${BASE}/arrange`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(req),
  });
  if (!res.ok) {
    throw await responseError(res, 'Arrange failed');
  }
  return res.json();
}

export async function generate(req: GenerateRequest): Promise<{ id: string; status: string }> {
  const res = await fetch(`${BASE}/generate`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(req),
  });
  if (!res.ok) {
    throw await responseError(res, 'Generate failed');
  }
  return res.json();
}

export async function getJob(id: string): Promise<JobStatus> {
  const res = await fetch(`${BASE}/jobs/${id}`);
  if (!res.ok) {
    throw await responseError(res, 'Job not found');
  }
  return res.json();
}

export async function getLibrary(status?: string): Promise<LibraryItem[]> {
  const params = status ? `?status=${status}` : '';
  const res = await fetch(`${BASE}/library${params}`);
  if (!res.ok) {
    throw await responseError(res, 'Library fetch failed');
  }
  return res.json();
}

export function audioUrl(filename: string): string {
  if (filename.startsWith('/api/')) return filename;
  if (filename.startsWith('/')) return `${BASE}${filename}`;
  const basename = filename.split(/[\\/]/).filter(Boolean).pop() || filename;
  return `${BASE}/audio/${encodeURIComponent(basename)}`;
}
