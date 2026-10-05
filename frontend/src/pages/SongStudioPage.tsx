/**
 * SongStudioPage — Song Studio UI (frontend/src/pages/SongStudioPage.tsx)
 * -------------------------------------------------------------------------
 * Full Song Studio interface:
 *   1. Upload song (mp3/wav/flac/m4a)
 *   2. Stage progress bar: Separate → Analyze → Convert → Remix → Master
 *   3. Key transpose control (±12 semitones)
 *   4. A/B comparison player (original vs cloned)
 *   5. Stems download (vocals, drums, bass, other, acapella, instrumental)
 *   6. Similarity score badge + quality report
 *   7. Consent legal gate
 *
 * Phase 8 will add the music-reactive theme tied to the uploaded song's features.
 * Heavy SVC/remaster work runs in a background job polled via /api/v1/song endpoints.
 */

import { useState, useRef, useCallback } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  Upload, Music, AudioLines, BarChart3, Download, Play, Pause,
  ArrowRight, CheckCircle, AlertTriangle, Loader2, Sliders, FileAudio,
} from "lucide-react";
import { API_BASE, getAuthToken } from "../services/api";

// ── Types ──────────────────────────────────────────────────────────────────────

type JobStage =
  | "idle"
  | "uploading"
  | "separating"
  | "analyzing"
  | "converting"
  | "remixing"
  | "mastering"
  | "done"
  | "error";

interface SongJobStatus {
  job_id: string;
  stage: string;
  progress: number;
  eta_seconds?: number;
  error?: string;
  result?: SongResult;
}

interface SongResult {
  full_song_wav?: string;
  full_song_mp3?: string;
  acapella_wav?: string;
  instrumental_wav?: string;
  similarity_score?: number;
  integrated_lufs?: number;
  true_peak_db?: number;
  musical_key?: string;
  bpm?: number;
  quality_passed?: boolean;
  stems?: Record<string, string>;
}

// ── Stage config ───────────────────────────────────────────────────────────────

const STAGES: { id: JobStage; label: string; icon: string }[] = [
  { id: "separating", label: "Separate Stems", icon: "🎛" },
  { id: "analyzing", label: "Analyze", icon: "📊" },
  { id: "converting", label: "Convert Voice", icon: "🎤" },
  { id: "remixing", label: "Remix", icon: "🎚" },
  { id: "mastering", label: "Master", icon: "🎯" },
];

const STAGE_ORDER: JobStage[] = [
  "uploading", "separating", "analyzing", "converting", "remixing", "mastering", "done",
];

function stageIndex(stage: JobStage): number {
  return STAGE_ORDER.indexOf(stage);
}

// ── Main Component ─────────────────────────────────────────────────────────────

export function SongStudioPage() {
  const [dragOver, setDragOver] = useState(false);
  const [songFile, setSongFile] = useState<File | null>(null);
  const [songId, setSongId] = useState<string | null>(null);
  const [jobStage, setJobStage] = useState<JobStage>("idle");
  const [progress, setProgress] = useState(0);
  const [etaSeconds, setEtaSeconds] = useState<number | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<SongResult | null>(null);
  const [pitchShift, setPitchShift] = useState(0);
  const [voiceProfileId, setVoiceProfileId] = useState<string>("");
  const [consentGiven, setConsentGiven] = useState(false);
  const [isPlayingOriginal, setIsPlayingOriginal] = useState(false);
  const [isPlayingCloned, setIsPlayingCloned] = useState(false);

  const fileInputRef = useRef<HTMLInputElement>(null);
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const originalAudioRef = useRef<HTMLAudioElement | null>(null);
  const clonedAudioRef = useRef<HTMLAudioElement | null>(null);

  // ── File Selection ─────────────────────────────────────────────────────────

  const handleFile = useCallback((file: File) => {
    const allowed = ["audio/mpeg", "audio/wav", "audio/flac", "audio/x-flac", "audio/mp4", "audio/x-m4a"];
    if (!allowed.some((t) => file.type.includes(t.split("/")[1]) || file.name.endsWith(".flac") || file.name.endsWith(".m4a") || file.name.endsWith(".mp3") || file.name.endsWith(".wav"))) {
      setError("Unsupported format. Please upload MP3, WAV, FLAC, or M4A.");
      return;
    }
    if (file.size > 200 * 1024 * 1024) {
      setError("File too large. Maximum 200 MB.");
      return;
    }
    setSongFile(file);
    setError(null);
    setResult(null);
    setJobStage("idle");
  }, []);

  const onDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setDragOver(false);
    const file = e.dataTransfer.files[0];
    if (file) handleFile(file);
  };

  // ── Upload & Start Pipeline ────────────────────────────────────────────────

  const startConversion = async () => {
    if (!songFile || !consentGiven) return;
    setError(null);
    setJobStage("uploading");
    setProgress(2);

    try {
      const token = getAuthToken();
      const formData = new FormData();
      formData.append("audio_file", songFile);

      // 1. Upload song
      const uploadRes = await fetch(`${API_BASE}/song/upload`, {
        method: "POST",
        headers: token ? { Authorization: `Bearer ${token}` } : {},
        body: formData,
      });

      if (!uploadRes.ok) {
        const err = await uploadRes.json().catch(() => ({}));
        throw new Error(err.detail || "Upload failed");
      }
      const { data: uploadData } = await uploadRes.json();
      const sid = uploadData.song_id;
      setSongId(sid);
      setJobStage("separating");
      setProgress(8);

      // 2. Start full conversion (async job)
      const convertRes = await fetch(`${API_BASE}/song/convert`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          ...(token ? { Authorization: `Bearer ${token}` } : {}),
        },
        body: JSON.stringify({
          song_id: sid,
          voice_profile_id: voiceProfileId || null,
          pitch_shift_semitones: pitchShift,
          user_consent: true,
          voice_profile_consent: true,
          async_job: true,
        }),
      });

      if (!convertRes.ok) {
        const err = await convertRes.json().catch(() => ({}));
        throw new Error(err.detail || "Conversion failed to start");
      }
      const { data: convertData } = await convertRes.json();
      const jobId = convertData.job_id;

      // 3. Poll for progress
      pollRef.current = setInterval(async () => {
        try {
          const statusRes = await fetch(`${API_BASE}/song/jobs/${jobId}`, {
            headers: token ? { Authorization: `Bearer ${token}` } : {},
          });
          if (!statusRes.ok) return;
          const { data: statusData }: { data: SongJobStatus } = await statusRes.json();

          setProgress(statusData.progress ?? progress);
          setEtaSeconds(statusData.eta_seconds ?? null);

          // Map backend stage string to our JobStage enum
          const stageMap: Record<string, JobStage> = {
            "stem_separation": "separating",
            "vocal_dereverb": "separating",
            "musical_analysis": "analyzing",
            "svc": "converting",
            "vocal_postprocess": "converting",
            "remixing": "remixing",
            "mastering": "mastering",
            "done": "done",
            "failed": "error",
          };
          const mappedStage = stageMap[statusData.stage] ?? jobStage;
          setJobStage(mappedStage);

          if (mappedStage === "done" || statusData.stage === "done") {
            clearInterval(pollRef.current!);
            setProgress(100);
            setJobStage("done");
            setResult(statusData.result ?? null);
          } else if (mappedStage === "error" || statusData.stage === "failed") {
            clearInterval(pollRef.current!);
            setJobStage("error");
            setError(statusData.error || "Processing failed");
          }
        } catch {
          // Polling errors are transient; keep trying
        }
      }, 2500);
    } catch (err: unknown) {
      setJobStage("error");
      setError(err instanceof Error ? err.message : "An error occurred");
    }
  };

  // ── Audio Playback ─────────────────────────────────────────────────────────

  const toggleOriginal = () => {
    if (!originalAudioRef.current) return;
    if (isPlayingOriginal) {
      originalAudioRef.current.pause();
      setIsPlayingOriginal(false);
    } else {
      clonedAudioRef.current?.pause();
      setIsPlayingCloned(false);
      originalAudioRef.current.play();
      setIsPlayingOriginal(true);
    }
  };

  const toggleCloned = () => {
    if (!clonedAudioRef.current) return;
    if (isPlayingCloned) {
      clonedAudioRef.current.pause();
      setIsPlayingCloned(false);
    } else {
      originalAudioRef.current?.pause();
      setIsPlayingOriginal(false);
      clonedAudioRef.current.play();
      setIsPlayingCloned(true);
    }
  };

  const downloadFile = (url: string, filename: string) => {
    const a = document.createElement("a");
    a.href = url;
    a.download = filename;
    a.click();
  };

  // ── Render ─────────────────────────────────────────────────────────────────

  const activeStageIdx = stageIndex(jobStage);
  const isProcessing = ["uploading", "separating", "analyzing", "converting", "remixing", "mastering"].includes(jobStage);

  return (
    <div
      className="min-h-screen pt-16"
      style={{ background: "linear-gradient(180deg, #000 0%, #050818 100%)" }}
    >
      <div className="max-w-4xl mx-auto px-4 sm:px-6 py-10 space-y-8">

        {/* ── Header ──────────────────────────────────────────────── */}
        <div>
          <h1 className="text-2xl sm:text-3xl font-semibold text-white tracking-tight">
            Song Studio
          </h1>
          <p className="text-sm mt-1" style={{ color: "rgba(255,255,255,0.5)" }}>
            Upload any song → get it sung in your voice, same music, same melody.
          </p>
        </div>

        {/* ── Legal Consent Gate ──────────────────────────────────── */}
        <div
          className="glass-card p-5 flex items-start gap-4"
        >
          <input
            id="song-consent"
            type="checkbox"
            checked={consentGiven}
            onChange={(e) => setConsentGiven(e.target.checked)}
            className="mt-0.5 w-5 h-5 accent-cyan-400 cursor-pointer flex-shrink-0"
            aria-label="Consent to song studio usage"
          />
          <label htmlFor="song-consent" className="text-sm cursor-pointer" style={{ color: "rgba(255,255,255,0.72)" }}>
            I confirm: (1) this is <strong className="text-white">my voice</strong> or I have explicit permission to use it; 
            (2) I have rights to upload this song for personal listening; 
            (3) I understand exports carry <em>"AI-generated voice"</em> metadata and this tool is not for sharing copyrighted songs publicly.
          </label>
        </div>

        {/* ── Upload Zone ─────────────────────────────────────────── */}
        <div
          className="glass-card relative overflow-hidden cursor-pointer transition-all duration-300"
          style={{
            border: dragOver
              ? "1px solid rgba(100,206,251,0.5)"
              : "1px solid rgba(255,255,255,0.08)",
            boxShadow: dragOver ? "0 0 24px rgba(100,206,251,0.15)" : "none",
          }}
          onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
          onDragLeave={() => setDragOver(false)}
          onDrop={onDrop}
          onClick={() => fileInputRef.current?.click()}
          role="button"
          aria-label="Upload song audio file"
          tabIndex={0}
          onKeyDown={(e) => e.key === "Enter" && fileInputRef.current?.click()}
        >
          <input
            ref={fileInputRef}
            type="file"
            accept="audio/*,.flac,.m4a"
            className="hidden"
            onChange={(e) => e.target.files?.[0] && handleFile(e.target.files[0])}
            aria-hidden="true"
          />
          <div className="flex flex-col items-center justify-center py-12 px-6 text-center gap-3">
            <motion.div
              animate={dragOver ? { scale: 1.15 } : { scale: 1 }}
              transition={{ type: "spring", stiffness: 300 }}
            >
              <Upload size={36} style={{ color: "rgba(100,206,251,0.7)" }} />
            </motion.div>
            {songFile ? (
              <div>
                <p className="text-white font-medium text-sm">{songFile.name}</p>
                <p className="text-xs mt-1" style={{ color: "rgba(255,255,255,0.4)" }}>
                  {(songFile.size / 1024 / 1024).toFixed(1)} MB — click to change
                </p>
              </div>
            ) : (
              <div>
                <p className="text-white text-sm font-medium">
                  Drop your song here or click to browse
                </p>
                <p className="text-xs mt-1" style={{ color: "rgba(255,255,255,0.4)" }}>
                  MP3, WAV, FLAC, M4A — up to 200 MB (≈10 min)
                </p>
              </div>
            )}
          </div>
        </div>

        {/* ── Settings Row ─────────────────────────────────────────── */}
        {songFile && (
          <div className="glass-card p-5 space-y-5">
            <h2 className="text-sm font-semibold text-white flex items-center gap-2">
              <Sliders size={15} /> Conversion Settings
            </h2>

            {/* Pitch shift */}
            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <label className="text-xs text-white/60" htmlFor="pitch-shift">
                  Key Transpose
                </label>
                <span className="text-xs font-mono text-white/80">
                  {pitchShift > 0 ? `+${pitchShift}` : pitchShift} semitones
                </span>
              </div>
              <input
                id="pitch-shift"
                type="range"
                min={-12}
                max={12}
                step={1}
                value={pitchShift}
                onChange={(e) => setPitchShift(Number(e.target.value))}
                className="w-full accent-cyan-400 cursor-pointer"
                aria-label="Key transpose in semitones"
              />
              <div className="flex justify-between text-xs" style={{ color: "rgba(255,255,255,0.3)" }}>
                <span>-12 (octave down)</span>
                <span>0 (original key)</span>
                <span>+12 (octave up)</span>
              </div>
            </div>

            {/* Voice profile ID (optional) */}
            <div className="space-y-2">
              <label className="text-xs text-white/60" htmlFor="voice-profile-input">
                Voice Profile ID <span className="text-white/30">(optional — leave blank for demo)</span>
              </label>
              <input
                id="voice-profile-input"
                type="text"
                placeholder="e.g. vp_abc123"
                value={voiceProfileId}
                onChange={(e) => setVoiceProfileId(e.target.value)}
                className="w-full px-4 py-2.5 rounded-xl text-sm text-white placeholder:text-white/25 outline-none transition-colors"
                style={{
                  background: "rgba(255,255,255,0.05)",
                  border: "1px solid rgba(255,255,255,0.1)",
                }}
              />
            </div>

            {/* Start button */}
            <button
              id="song-studio-convert-btn"
              onClick={startConversion}
              disabled={!consentGiven || isProcessing}
              className="w-full flex items-center justify-center gap-3 py-3.5 rounded-xl text-sm font-semibold text-white transition-all duration-300 disabled:opacity-40 disabled:cursor-not-allowed"
              style={{
                background: "linear-gradient(135deg, rgba(100,206,251,0.18), rgba(167,139,250,0.18))",
                border: "1px solid rgba(100,206,251,0.3)",
              }}
              aria-label="Start voice conversion"
            >
              {isProcessing ? (
                <><Loader2 size={16} className="animate-spin" /> Processing…</>
              ) : (
                <><Music size={16} /> Convert to My Voice <ArrowRight size={15} /></>
              )}
            </button>
          </div>
        )}

        {/* ── Stage Progress ───────────────────────────────────────── */}
        <AnimatePresence>
          {isProcessing && (
            <motion.div
              className="glass-card p-6 space-y-5"
              initial={{ opacity: 0, y: 12 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0 }}
            >
              {/* Stage timeline */}
              <div className="flex items-center gap-2 overflow-x-auto scrollbar-hide pb-1">
                {STAGES.map((stage, idx) => {
                  const stagePos = STAGE_ORDER.indexOf(stage.id);
                  const isCompleted = stagePos < activeStageIdx;
                  const isCurrent = stage.id === jobStage;
                  return (
                    <div key={stage.id} className="flex items-center gap-2 flex-shrink-0">
                      <div
                        className="flex items-center gap-2 px-3 py-2 rounded-full text-xs font-medium transition-all duration-500"
                        style={{
                          background: isCurrent
                            ? "rgba(100,206,251,0.15)"
                            : isCompleted
                            ? "rgba(52,211,153,0.12)"
                            : "rgba(255,255,255,0.04)",
                          border: isCurrent
                            ? "1px solid rgba(100,206,251,0.4)"
                            : isCompleted
                            ? "1px solid rgba(52,211,153,0.3)"
                            : "1px solid rgba(255,255,255,0.07)",
                          color: isCurrent ? "#64CEFB" : isCompleted ? "#34d399" : "rgba(255,255,255,0.35)",
                        }}
                      >
                        {isCompleted ? <CheckCircle size={12} /> : isCurrent ? <Loader2 size={12} className="animate-spin" /> : <span>{stage.icon}</span>}
                        {stage.label}
                      </div>
                      {idx < STAGES.length - 1 && (
                        <div
                          className="h-px w-4 flex-shrink-0 transition-colors duration-500"
                          style={{
                            background: stagePos < activeStageIdx
                              ? "rgba(52,211,153,0.4)"
                              : "rgba(255,255,255,0.08)",
                          }}
                        />
                      )}
                    </div>
                  );
                })}
              </div>

              {/* Progress bar */}
              <div className="space-y-1.5">
                <div className="flex justify-between text-xs" style={{ color: "rgba(255,255,255,0.5)" }}>
                  <span>{progress}% complete</span>
                  {etaSeconds != null && <span>ETA ~{Math.round(etaSeconds)}s</span>}
                </div>
                <div className="h-1.5 rounded-full overflow-hidden" style={{ background: "rgba(255,255,255,0.08)" }}>
                  <motion.div
                    className="h-full rounded-full"
                    style={{ background: "linear-gradient(90deg, #64CEFB, #a78bfa)" }}
                    animate={{ width: `${progress}%` }}
                    transition={{ duration: 0.5, ease: "easeOut" }}
                  />
                </div>
              </div>
            </motion.div>
          )}
        </AnimatePresence>

        {/* ── Error ───────────────────────────────────────────────── */}
        <AnimatePresence>
          {error && (
            <motion.div
              className="glass-card p-4 flex items-start gap-3"
              style={{ borderColor: "rgba(239,68,68,0.3)", background: "rgba(239,68,68,0.08)" }}
              initial={{ opacity: 0, y: 8 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0 }}
              role="alert"
            >
              <AlertTriangle size={16} className="flex-shrink-0 mt-0.5" style={{ color: "#f87171" }} />
              <p className="text-sm" style={{ color: "#fca5a5" }}>{error}</p>
            </motion.div>
          )}
        </AnimatePresence>

        {/* ── Results ──────────────────────────────────────────────── */}
        <AnimatePresence>
          {result && jobStage === "done" && (
            <motion.div
              className="space-y-6"
              initial={{ opacity: 0, y: 16 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5 }}
            >
              {/* Metadata bar */}
              <div className="glass-card p-5 flex flex-wrap gap-4">
                {result.musical_key && (
                  <div className="text-center">
                    <p className="text-xs text-white/40">Key</p>
                    <p className="text-sm font-semibold text-white">{result.musical_key}</p>
                  </div>
                )}
                {result.bpm && (
                  <div className="text-center">
                    <p className="text-xs text-white/40">BPM</p>
                    <p className="text-sm font-semibold text-white">{Math.round(result.bpm)}</p>
                  </div>
                )}
                {result.integrated_lufs != null && (
                  <div className="text-center">
                    <p className="text-xs text-white/40">Loudness</p>
                    <p className="text-sm font-semibold text-white">{result.integrated_lufs.toFixed(1)} LUFS</p>
                  </div>
                )}
                {result.similarity_score != null && (
                  <div className="text-center">
                    <p className="text-xs text-white/40">Voice Match</p>
                    <p
                      className="text-sm font-semibold"
                      style={{ color: result.similarity_score >= 0.85 ? "#34d399" : "#f59e0b" }}
                    >
                      {Math.round(result.similarity_score * 100)}%
                    </p>
                  </div>
                )}
                <div className="flex items-center gap-1.5 ml-auto">
                  {result.quality_passed ? (
                    <CheckCircle size={14} className="text-green-400" />
                  ) : (
                    <AlertTriangle size={14} className="text-amber-400" />
                  )}
                  <span className="text-xs text-white/60">
                    {result.quality_passed ? "Quality gate passed" : "Quality warnings present"}
                  </span>
                </div>
              </div>

              {/* A/B Comparison Player */}
              <div className="glass-card p-5 space-y-4">
                <h3 className="text-sm font-semibold text-white flex items-center gap-2">
                  <BarChart3 size={15} /> A/B Comparison
                </h3>

                {/* Original track player */}
                <div
                  className="flex items-center gap-4 p-4 rounded-xl"
                  style={{ background: "rgba(255,255,255,0.03)", border: "1px solid rgba(255,255,255,0.07)" }}
                >
                  <button
                    onClick={toggleOriginal}
                    className="flex items-center justify-center w-10 h-10 rounded-full transition-all"
                    style={{
                      background: isPlayingOriginal ? "rgba(255,255,255,0.15)" : "rgba(255,255,255,0.07)",
                      minHeight: 44,
                      minWidth: 44,
                    }}
                    aria-label={isPlayingOriginal ? "Pause original" : "Play original"}
                  >
                    {isPlayingOriginal ? <Pause size={16} className="text-white" /> : <Play size={16} className="text-white" />}
                  </button>
                  <div>
                    <p className="text-sm font-medium text-white">A — Original Song</p>
                    <p className="text-xs text-white/40">Original uploaded audio</p>
                  </div>
                  {songFile && (
                    <audio
                      ref={originalAudioRef}
                      src={URL.createObjectURL(songFile)}
                      onEnded={() => setIsPlayingOriginal(false)}
                      preload="none"
                    />
                  )}
                </div>

                {/* Cloned track player */}
                <div
                  className="flex items-center gap-4 p-4 rounded-xl"
                  style={{ background: "rgba(100,206,251,0.04)", border: "1px solid rgba(100,206,251,0.15)" }}
                >
                  <button
                    onClick={toggleCloned}
                    className="flex items-center justify-center w-10 h-10 rounded-full transition-all"
                    style={{
                      background: isPlayingCloned ? "rgba(100,206,251,0.25)" : "rgba(100,206,251,0.1)",
                      minHeight: 44,
                      minWidth: 44,
                    }}
                    aria-label={isPlayingCloned ? "Pause cloned" : "Play cloned"}
                  >
                    {isPlayingCloned ? <Pause size={16} style={{ color: "#64CEFB" }} /> : <Play size={16} style={{ color: "#64CEFB" }} />}
                  </button>
                  <div>
                    <p className="text-sm font-medium" style={{ color: "#64CEFB" }}>B — My Voice Version</p>
                    <p className="text-xs text-white/40">Lead vocal replaced with your cloned voice</p>
                  </div>
                  {result.full_song_mp3 && (
                    <audio
                      ref={clonedAudioRef}
                      src={`${API_BASE}${result.full_song_mp3}`}
                      onEnded={() => setIsPlayingCloned(false)}
                      preload="none"
                    />
                  )}
                </div>
              </div>

              {/* Downloads */}
              <div className="glass-card p-5 space-y-3">
                <h3 className="text-sm font-semibold text-white flex items-center gap-2">
                  <Download size={15} /> Downloads
                </h3>
                <div className="grid grid-cols-2 sm:grid-cols-3 gap-3">
                  {[
                    { key: "full_song_wav", label: "Full Song (WAV)", icon: "🎵" },
                    { key: "full_song_mp3", label: "Full Song (MP3)", icon: "🎵" },
                    { key: "acapella_wav", label: "My Voice Acapella", icon: "🎤" },
                    { key: "instrumental_wav", label: "Instrumental", icon: "🎸" },
                  ].map(({ key, label, icon }) => {
                    const url = result[key as keyof SongResult] as string | undefined;
                    if (!url) return null;
                    return (
                      <button
                        key={key}
                        onClick={() => downloadFile(`${API_BASE}${url}`, label.replace(/[^a-z0-9]/gi, "_").toLowerCase() + ".wav")}
                        className="flex items-center gap-2 px-4 py-3 rounded-xl text-xs font-medium text-white/80 hover:text-white transition-all"
                        style={{
                          background: "rgba(255,255,255,0.04)",
                          border: "1px solid rgba(255,255,255,0.08)",
                          minHeight: 44,
                        }}
                      >
                        <span>{icon}</span>
                        <FileAudio size={12} />
                        <span className="truncate">{label}</span>
                      </button>
                    );
                  })}
                </div>
              </div>

              {/* Stems */}
              {result.stems && Object.keys(result.stems).length > 0 && (
                <div className="glass-card p-5 space-y-3">
                  <h3 className="text-sm font-semibold text-white flex items-center gap-2">
                    <AudioLines size={15} /> Individual Stems
                  </h3>
                  <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
                    {Object.entries(result.stems).map(([stem, url]) => (
                      <button
                        key={stem}
                        onClick={() => downloadFile(`${API_BASE}${url}`, `${stem}.wav`)}
                        className="flex items-center gap-2 px-3 py-2.5 rounded-xl text-xs font-medium text-white/70 hover:text-white transition-all capitalize"
                        style={{
                          background: "rgba(255,255,255,0.04)",
                          border: "1px solid rgba(255,255,255,0.08)",
                          minHeight: 44,
                        }}
                      >
                        <Download size={11} /> {stem}
                      </button>
                    ))}
                  </div>
                </div>
              )}
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    </div>
  );
}
