/**
 * EchoVoice Backend API Client (frontend/src/services/api.ts)
 * ------------------------------------------------------------
 * Handles communication with FastAPI backend:
 * - Authentication & Session Management (Login, Register, Demo Auto-Login, /auth/me)
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

export interface UserProfile {
  id: string;
  email: string;
  full_name?: string;
  is_active: boolean;
  is_superuser: boolean;
  created_at: string;
  updated_at: string;
}

export const API_BASE = "http://localhost:8000/api/v1";
export const TOKEN_KEY = "echovoice_token";

// -------------------------------------------------------------
// Authentication & Session Management
// -------------------------------------------------------------

export function getAuthToken(): string | null {
  return localStorage.getItem(TOKEN_KEY);
}

export function setAuthToken(token: string): void {
  localStorage.setItem(TOKEN_KEY, token);
}

export function clearAuthToken(): void {
  localStorage.removeItem(TOKEN_KEY);
}

export async function loginUser(email: string, password: string): Promise<string> {
  const res = await fetch(`${API_BASE}/auth/login`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ email, password }),
  });

  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || "Authentication failed. Invalid email or password.");
  }

  const json = await res.json();
  const token = json.data.access_token;
  setAuthToken(token);
  return token;
}

export async function registerUser(email: string, password: string, fullName?: string): Promise<string> {
  const res = await fetch(`${API_BASE}/auth/register`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      email,
      password,
      full_name: fullName || "Studio Artist",
    }),
  });

  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || "Registration failed.");
  }

  const json = await res.json();
  const token = json.data.access_token;
  setAuthToken(token);
  return token;
}

export async function fetchCurrentUser(): Promise<UserProfile | null> {
  const token = getAuthToken();
  if (!token) return null;

  try {
    const res = await fetch(`${API_BASE}/auth/me`, {
      method: "GET",
      headers: {
        Authorization: `Bearer ${token}`,
      },
    });

    if (!res.ok) {
      if (res.status === 401) {
        clearAuthToken();
      }
      return null;
    }

    const json = await res.json();
    return json.data;
  } catch {
    return null;
  }
}

/**
 * Ensures an active authenticated session exists.
 * If no session is found, transparently logs in as the default studio demo user.
 */
export async function ensureAuthenticatedSession(): Promise<string> {
  const existing = getAuthToken();
  if (existing) {
    const me = await fetchCurrentUser();
    if (me) return existing;
  }

  // Auto-connect as demo user
  try {
    return await loginUser("demo@echovoice.ai", "DemoPassword123!");
  } catch {
    return await registerUser("demo@echovoice.ai", "DemoPassword123!", "EchoVoice Demo");
  }
}

/**
 * Retrieves valid Bearer token, auto-bootstrapping if none is set.
 */
export async function getValidAuthToken(): Promise<string> {
  const token = getAuthToken();
  if (token) return token;
  return await ensureAuthenticatedSession();
}

// -------------------------------------------------------------
// Voice Profiles & Quality Analysis
// -------------------------------------------------------------

function getSafeAudioFilename(file: File | Blob): string {
  if (file instanceof File && file.name) {
    return file.name;
  }
  const type = file.type || "";
  if (type.includes("webm")) return "recording.webm";
  if (type.includes("ogg")) return "recording.ogg";
  if (type.includes("mp4") || type.includes("m4a")) return "recording.m4a";
  if (type.includes("mp3")) return "recording.mp3";
  return "recording.wav";
}

export async function analyzeAudioQuality(file: File | Blob): Promise<AudioQualityMetrics> {
  const formData = new FormData();
  formData.append("file", file, getSafeAudioFilename(file));

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
  formData.append("file", file, getSafeAudioFilename(file));
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
  const token = await getValidAuthToken();
  const formData = new FormData();
  formData.append("file", file, getSafeAudioFilename(file));
  formData.append("name", name);
  formData.append("description", description);
  formData.append("consent_given", String(consentGiven));

  const headers: HeadersInit = {
    Authorization: `Bearer ${token}`,
  };

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
  try {
    const token = await getValidAuthToken();
    const headers: HeadersInit = {
      Authorization: `Bearer ${token}`,
    };

    const res = await fetch(`${API_BASE}/voice-profiles/`, {
      method: "GET",
      headers,
    });

    if (!res.ok) {
      return [];
    }

    const json = await res.json();
    return json.data || [];
  } catch {
    return [];
  }
}

export async function deleteVoiceProfile(profileId: string): Promise<boolean> {
  try {
    const token = await getValidAuthToken();
    const headers: HeadersInit = {
      Authorization: `Bearer ${token}`,
    };

    const res = await fetch(`${API_BASE}/voice-profiles/${profileId}`, {
      method: "DELETE",
      headers,
    });
    return res.ok;
  } catch {
    return false;
  }
}

export async function setDefaultVoiceProfile(profileId: string): Promise<VoiceProfile> {
  const token = await getValidAuthToken();
  const headers: HeadersInit = {
    Authorization: `Bearer ${token}`,
  };

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
  const token = await getValidAuthToken();
  const headers: HeadersInit = {
    "Content-Type": "application/json",
    Authorization: `Bearer ${token}`,
  };

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
  const token = await getValidAuthToken();
  const headers: HeadersInit = {
    Authorization: `Bearer ${token}`,
  };

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
  try {
    const token = await getValidAuthToken();
    const headers: HeadersInit = {
      Authorization: `Bearer ${token}`,
    };

    const res = await fetch(`${API_BASE}/tts-jobs`, {
      method: "GET",
      headers,
    });

    if (!res.ok) {
      return [];
    }

    const json = await res.json();
    return json.data || [];
  } catch {
    return [];
  }
}

export async function resumeTTSJob(jobId: string): Promise<TTSJob> {
  const token = await getValidAuthToken();
  const headers: HeadersInit = {
    Authorization: `Bearer ${token}`,
  };

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
  const token = await getValidAuthToken();
  const headers: HeadersInit = {
    Authorization: `Bearer ${token}`,
  };

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
