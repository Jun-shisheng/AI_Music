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
}

export interface JobStatus {
  id: string;
  title: string;
  status: 'queued' | 'running' | 'succeeded' | 'failed';
  lyrics?: string;
  style_prompt?: string;
  instrumental?: boolean;
  audio_path?: string;
  audio_url?: string;
  error?: string;
  created_at: string;
  updated_at: string;
}

export interface LibraryItem extends JobStatus {}

export async function arrange(req: ArrangeRequest): Promise<ArrangeResponse> {
  const res = await fetch(`${BASE}/arrange`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(req),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || 'Arrange failed');
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
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || 'Generate failed');
  }
  return res.json();
}

export async function getJob(id: string): Promise<JobStatus> {
  const res = await fetch(`${BASE}/jobs/${id}`);
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || 'Job not found');
  }
  return res.json();
}

export async function getLibrary(status?: string): Promise<LibraryItem[]> {
  const params = status ? `?status=${status}` : '';
  const res = await fetch(`${BASE}/library${params}`);
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || 'Library fetch failed');
  }
  return res.json();
}

export function audioUrl(filename: string): string {
  if (filename.startsWith('/api/')) return filename;
  if (filename.startsWith('/')) return `${BASE}${filename}`;
  const basename = filename.split(/[\\/]/).filter(Boolean).pop() || filename;
  return `${BASE}/audio/${encodeURIComponent(basename)}`;
}
