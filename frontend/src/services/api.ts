/**
 * EchoVoice Backend API Client (frontend/src/services/api.ts)
 * ------------------------------------------------------------
 * Handles communication with FastAPI backend:
 * - Voice Profile creation, audio quality analysis, and restoration preview
 * - User voice profile library (list, preview, set default, delete)
 * - Real-time TTS synthesis (< 250 characters)
 * - Long-form job submission, status polling, pause/resume, and export download
 */

export interface AudioQualityMetrics {
  snr_db: number;
  clipping_ratio: number;
  speech_duration_seconds: number;
  speech_ratio: number;
  noise_floor_db: number;
  is_usable: boolean;
  recommendation: string;
}

export interface VoiceProfile {
  id: string;
  name: string;
  description?: string;
  reference_audio_path: string;
  consent_given: boolean;
  consent_timestamp?: string;
  quality_metrics?: AudioQualityMetrics;
  is_default: boolean;
  created_at: string;
}

export interface CleanPreviewResult {
  cleaned_audio_url: string;
  summary: {
    sample_rate: number;
    raw_snr_db: number;
    cleaned_snr_db: number;
    snr_improvement_db: number;
    raw_noise_floor_db: number;
    cleaned_noise_floor_db: number;
    cleaned_duration_sec: number;
    steps_applied: string[];
    is_usable: boolean;
    recommendation: string;
  };
  raw_metrics: AudioQualityMetrics;
  cleaned_metrics: AudioQualityMetrics;
}

export interface SynthesisResult {
  download_url: string;
  filename: string;
  duration_seconds: number;
  latency_ms: number;
  rtf: number;
  engine: string;
  language: string;
}

export interface TTSJob {
  id: string;
  user_id: string;
  voice_profile_id?: string;
  status: "PENDING" | "PROCESSING" | "COMPLETED" | "FAILED" | "CANCELLED";
  target_language: string;
  engine: string;
  mastering_preset: string;
  progress: number;
  total_chunks: number;
  completed_chunks: number;
  output_wav_path?: string;
  output_mp3_path?: string;
  output_m4a_path?: string;
  wer_score?: number;
  similarity_score?: number;
  audio_report?: {
    integrated_lufs?: number;
    true_peak_db?: number;
    duration_seconds?: number;
    snr_db?: number;
    clipping_ratio?: number;
    qc_passed?: boolean;
    qc_warnings?: string[];
    steps_applied?: string[];
  };
  error_message?: string;
  created_at: string;
  updated_at: string;
}

export const API_BASE = "http://localhost:8000/api/v1";

// Auth token storage helper
export function getAuthToken(): string | null {
  return localStorage.getItem("echovoice_token");
}

export function setAuthToken(token: string): void {
  localStorage.setItem("echovoice_token", token);
}

// -------------------------------------------------------------
// Voice Profiles & Quality Analysis
// -------------------------------------------------------------

export async function analyzeAudioQuality(file: File | Blob): Promise<AudioQualityMetrics> {
  const formData = new FormData();
  formData.append("file", file, "recording.wav");

  const res = await fetch(`${API_BASE}/voice-profiles/analyze-quality`, {
    method: "POST",
    body: formData,
  });

  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || "Failed to analyze audio quality");
  }

  const json = await res.json();
  return json.data;
}

export async function previewCleanAudio(
  file: File | Blob,
  options?: { enableDenoise?: boolean; enableVadTrim?: boolean; targetLufs?: number }
): Promise<CleanPreviewResult> {
  const formData = new FormData();
  formData.append("file", file, "recording.wav");
  formData.append("enable_denoise", String(options?.enableDenoise ?? true));
  formData.append("enable_vad_trim", String(options?.enableVadTrim ?? true));
  formData.append("target_lufs", String(options?.targetLufs ?? -14.0));

  const res = await fetch(`${API_BASE}/voice-profiles/preview-clean`, {
    method: "POST",
    body: formData,
  });

  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || "Failed to clean audio preview");
  }

  const json = await res.json();
  return json.data;
}

export async function enrollVoiceProfile(
  file: File | Blob,
  name: string,
  description: string,
  consentGiven: boolean
): Promise<VoiceProfile> {
  const formData = new FormData();
  formData.append("file", file, "recording.wav");
  formData.append("name", name);
  formData.append("description", description);
  formData.append("consent_given", String(consentGiven));

  const headers: HeadersInit = {};
  const token = getAuthToken();
  if (token) {
    headers["Authorization"] = `Bearer ${token}`;
  }

  const res = await fetch(`${API_BASE}/voice-profiles/enroll`, {
    method: "POST",
    headers,
    body: formData,
  });

  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || "Failed to enroll voice profile");
  }

  const json = await res.json();
  return json.data;
}

export async function fetchUserProfiles(): Promise<VoiceProfile[]> {
  const headers: HeadersInit = {};
  const token = getAuthToken();
  if (token) {
    headers["Authorization"] = `Bearer ${token}`;
  }

  const res = await fetch(`${API_BASE}/voice-profiles/`, {
    method: "GET",
    headers,
  });

  if (!res.ok) {
    return [];
  }

  const json = await res.json();
  return json.data || [];
}

export async function deleteVoiceProfile(profileId: string): Promise<boolean> {
  const headers: HeadersInit = {};
  const token = getAuthToken();
  if (token) {
    headers["Authorization"] = `Bearer ${token}`;
  }

  const res = await fetch(`${API_BASE}/voice-profiles/${profileId}`, {
    method: "DELETE",
    headers,
  });
  return res.ok;
}

export async function setDefaultVoiceProfile(profileId: string): Promise<VoiceProfile> {
  const headers: HeadersInit = {};
  const token = getAuthToken();
  if (token) {
    headers["Authorization"] = `Bearer ${token}`;
  }

  const res = await fetch(`${API_BASE}/voice-profiles/${profileId}/default`, {
    method: "POST",
    headers,
  });
  const json = await res.json();
  return json.data;
}

// -------------------------------------------------------------
// Real-time TTS & Long-Form Jobs
// -------------------------------------------------------------

export async function synthesizeSpeech(params: {
  text: string;
  speaker_wav_path?: string;
  language?: string;
  speed?: number;
  engine?: string;
}): Promise<SynthesisResult> {
  const res = await fetch(`${API_BASE}/tts/synthesize`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      text: params.text,
      speaker_wav_path: params.speaker_wav_path,
      language: params.language || "en",
      speed: params.speed || 1.0,
      engine: params.engine || "xtts_v2",
    }),
  });

  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || "Speech synthesis failed");
  }

  const json = await res.json();
  return json.data;
}

export async function submitTTSJob(params: {
  script_text: string;
  voice_profile_id?: string;
  target_language?: string;
  engine?: string;
  mastering_preset?: string;
}): Promise<TTSJob> {
  const headers: HeadersInit = { "Content-Type": "application/json" };
  const token = getAuthToken();
  if (token) {
    headers["Authorization"] = `Bearer ${token}`;
  }

  const res = await fetch(`${API_BASE}/tts-jobs`, {
    method: "POST",
    headers,
    body: JSON.stringify({
      script_text: params.script_text,
      voice_profile_id: params.voice_profile_id,
      target_language: params.target_language || "en",
      engine: params.engine || "xtts_v2",
      mastering_preset: params.mastering_preset || "youtube_voiceover",
    }),
  });

  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || "Failed to schedule speech generation job");
  }

  const json = await res.json();
  return json.data;
}

export async function getTTSJob(jobId: string): Promise<TTSJob> {
  const headers: HeadersInit = {};
  const token = getAuthToken();
  if (token) {
    headers["Authorization"] = `Bearer ${token}`;
  }

  const res = await fetch(`${API_BASE}/tts-jobs/${jobId}`, {
    method: "GET",
    headers,
  });

  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || "Failed to fetch job status");
  }

  const json = await res.json();
  return json.data;
}

export async function listTTSJobs(): Promise<TTSJob[]> {
  const headers: HeadersInit = {};
  const token = getAuthToken();
  if (token) {
    headers["Authorization"] = `Bearer ${token}`;
  }

  const res = await fetch(`${API_BASE}/tts-jobs`, {
    method: "GET",
    headers,
  });

  if (!res.ok) {
    return [];
  }

  const json = await res.json();
  return json.data || [];
}

export async function resumeTTSJob(jobId: string): Promise<TTSJob> {
  const headers: HeadersInit = {};
  const token = getAuthToken();
  if (token) {
    headers["Authorization"] = `Bearer ${token}`;
  }

  const res = await fetch(`${API_BASE}/tts-jobs/${jobId}/resume`, {
    method: "POST",
    headers,
  });

  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || "Failed to resume job");
  }

  const json = await res.json();
  return json.data;
}

export async function cancelTTSJob(jobId: string): Promise<TTSJob> {
  const headers: HeadersInit = {};
  const token = getAuthToken();
  if (token) {
    headers["Authorization"] = `Bearer ${token}`;
  }

  const res = await fetch(`${API_BASE}/tts-jobs/${jobId}/cancel`, {
    method: "POST",
    headers,
  });

  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || "Failed to cancel job");
  }

  const json = await res.json();
  return json.data;
}
