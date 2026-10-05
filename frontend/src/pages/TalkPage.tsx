/**
 * TalkPage — Real-Time Conversational Voice Agent (frontend/src/pages/TalkPage.tsx)
 * ----------------------------------------------------------------------------------
 * Full-featured voice agent page with orb/waveform, listening/thinking/speaking
 * states, streaming transcript, citation chips, barge-in button, and hands-free toggle.
 * Phase 9 will wire the actual WebSocket realtime loop; this page provides the complete
 * shell with all UI states implemented and ready for backend connection.
 */

import { useState, useRef, useCallback } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Mic, MicOff, Volume2, Zap, BookOpen, RotateCcw } from "lucide-react";

type AgentState = "idle" | "listening" | "thinking" | "speaking";

interface Citation {
  id: string;
  title: string;
  snippet: string;
  score: number;
}

interface TranscriptEntry {
  role: "user" | "agent";
  text: string;
  citations?: Citation[];
  timestamp: Date;
}

export function TalkPage() {
  const [agentState, setAgentState] = useState<AgentState>("idle");
  const [handsFree, setHandsFree] = useState(false);
  const [transcript, setTranscript] = useState<TranscriptEntry[]>([]);
  const [isBargeInActive, setIsBargeInActive] = useState(false);
  const transcriptRef = useRef<HTMLDivElement>(null);

  const stateConfig: Record<AgentState, { label: string; color: string; pulseColor: string }> = {
    idle: { label: "Tap to speak", color: "#64CEFB", pulseColor: "rgba(100,206,251,0.2)" },
    listening: { label: "Listening…", color: "#34d399", pulseColor: "rgba(52,211,153,0.25)" },
    thinking: { label: "Thinking…", color: "#a78bfa", pulseColor: "rgba(167,139,250,0.25)" },
    speaking: { label: "Speaking…", color: "#f59e0b", pulseColor: "rgba(245,158,11,0.25)" },
  };

  const { label, color, pulseColor } = stateConfig[agentState];

  const handleOrbClick = useCallback(() => {
    if (agentState === "idle") {
      setAgentState("listening");
      // Phase 9: open WebSocket, start VAD
    } else if (agentState === "listening") {
      setAgentState("thinking");
      // Phase 9: send audio buffer to backend
      setTimeout(() => {
        setTranscript((prev) => [
          ...prev,
          {
            role: "user",
            text: "Tell me about voice cloning technology.",
            timestamp: new Date(),
          },
        ]);
        setAgentState("speaking");
        setTimeout(() => {
          setTranscript((prev) => [
            ...prev,
            {
              role: "agent",
              text: "Voice cloning technology uses neural networks to learn the unique characteristics of a person's voice from a short audio sample, then reproduce any text in that voice with high fidelity.",
              citations: [
                { id: "c1", title: "XTTS-v2 Paper", snippet: "Zero-shot cross-lingual synthesis…", score: 0.92 },
                { id: "c2", title: "RVC v2 Guide", snippet: "HuBERT + RMVPE pitch extraction…", score: 0.88 },
              ],
              timestamp: new Date(),
            },
          ]);
          setAgentState("idle");
        }, 3000);
      }, 1500);
    } else if (agentState === "speaking") {
      // Barge-in
      setAgentState("listening");
    }
  }, [agentState]);

  const handleBargeIn = () => {
    if (agentState === "speaking") {
      setIsBargeInActive(true);
      setAgentState("listening");
      setTimeout(() => setIsBargeInActive(false), 500);
    }
  };

  return (
    <div
      className="min-h-screen pt-16 flex flex-col"
      style={{ background: "linear-gradient(180deg, #000 0%, #050818 100%)" }}
    >
      <div className="flex-1 max-w-4xl mx-auto w-full px-4 sm:px-6 py-8 flex flex-col gap-8">
        {/* ── Page Header ──────────────────────────────────────────── */}
        <div>
          <h1 className="text-2xl sm:text-3xl font-semibold text-white tracking-tight">
            Voice Agent
          </h1>
          <p className="text-sm text-white/50 mt-1">
            Conversational AI with RAG knowledge, streaming responses, and voice cloning.
          </p>
        </div>

        {/* ── Agent Orb ────────────────────────────────────────────── */}
        <div className="flex flex-col items-center gap-6 py-8">
          {/* Pulse rings */}
          <div className="relative flex items-center justify-center" style={{ width: 200, height: 200 }}>
            {agentState !== "idle" && (
              <>
                <motion.div
                  className="absolute rounded-full"
                  style={{ background: pulseColor }}
                  animate={{ scale: [1, 1.5, 1], opacity: [0.6, 0, 0.6] }}
                  transition={{ duration: 2, repeat: Infinity, ease: "easeInOut" }}
                  style={{ width: 200, height: 200, background: pulseColor }}
                />
                <motion.div
                  className="absolute rounded-full"
                  animate={{ scale: [1, 1.3, 1], opacity: [0.4, 0, 0.4] }}
                  transition={{ duration: 2, repeat: Infinity, ease: "easeInOut", delay: 0.4 }}
                  style={{ width: 200, height: 200, background: pulseColor }}
                />
              </>
            )}

            {/* Main orb */}
            <motion.button
              onClick={handleOrbClick}
              className="relative rounded-full flex items-center justify-center cursor-pointer focus-visible:outline-none"
              style={{
                width: 120,
                height: 120,
                background: `radial-gradient(circle at 35% 35%, ${color}33, ${color}11)`,
                border: `2px solid ${color}55`,
                boxShadow: agentState !== "idle" ? `0 0 40px ${color}44` : "none",
                minHeight: 44,
                minWidth: 44,
              }}
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.96 }}
              animate={
                agentState === "listening"
                  ? { scale: [1, 1.04, 1] }
                  : agentState === "thinking"
                  ? { rotate: [0, 360] }
                  : {}
              }
              transition={
                agentState === "listening"
                  ? { duration: 0.8, repeat: Infinity }
                  : agentState === "thinking"
                  ? { duration: 2, repeat: Infinity, ease: "linear" }
                  : {}
              }
              aria-label={`Voice agent: ${label}`}
            >
              {agentState === "idle" && <Mic size={36} color={color} />}
              {agentState === "listening" && <Mic size={36} color={color} />}
              {agentState === "thinking" && (
                <motion.div
                  animate={{ rotate: 360 }}
                  transition={{ duration: 1.5, repeat: Infinity, ease: "linear" }}
                >
                  <Zap size={36} color={color} />
                </motion.div>
              )}
              {agentState === "speaking" && <Volume2 size={36} color={color} />}
            </motion.button>
          </div>

          <p className="text-sm font-medium" style={{ color }}>
            {label}
          </p>

          {/* Controls */}
          <div className="flex items-center gap-3">
            {/* Barge-in */}
            <button
              onClick={handleBargeIn}
              disabled={agentState !== "speaking"}
              className="flex items-center gap-2 px-4 py-2.5 rounded-full text-xs font-medium transition-all duration-200 disabled:opacity-30 disabled:cursor-not-allowed"
              style={{
                background: isBargeInActive ? "rgba(239,68,68,0.2)" : "rgba(255,255,255,0.06)",
                border: "1px solid rgba(255,255,255,0.12)",
                color: "rgba(255,255,255,0.7)",
                minHeight: 44,
              }}
              id="barge-in-btn"
              aria-label="Interrupt agent (barge-in)"
            >
              <MicOff size={13} /> Interrupt
            </button>

            {/* Hands-free toggle */}
            <button
              onClick={() => setHandsFree((v) => !v)}
              className="flex items-center gap-2 px-4 py-2.5 rounded-full text-xs font-medium transition-all duration-200"
              style={{
                background: handsFree ? "rgba(100,206,251,0.15)" : "rgba(255,255,255,0.06)",
                border: handsFree
                  ? "1px solid rgba(100,206,251,0.35)"
                  : "1px solid rgba(255,255,255,0.12)",
                color: handsFree ? "#64CEFB" : "rgba(255,255,255,0.7)",
                minHeight: 44,
              }}
              aria-pressed={handsFree}
              id="hands-free-toggle"
              aria-label={handsFree ? "Disable hands-free mode" : "Enable hands-free mode"}
            >
              <Mic size={13} /> Hands-free {handsFree ? "ON" : "OFF"}
            </button>

            {/* Reset */}
            <button
              onClick={() => { setTranscript([]); setAgentState("idle"); }}
              className="flex items-center justify-center w-10 h-10 rounded-full transition-colors duration-200"
              style={{
                background: "rgba(255,255,255,0.06)",
                border: "1px solid rgba(255,255,255,0.1)",
                color: "rgba(255,255,255,0.5)",
                minHeight: 44,
                minWidth: 44,
              }}
              aria-label="Reset conversation"
            >
              <RotateCcw size={14} />
            </button>
          </div>
        </div>

        {/* ── Transcript ───────────────────────────────────────────── */}
        <div
          ref={transcriptRef}
          className="flex-1 space-y-4 max-h-96 overflow-y-auto scrollbar-hide"
        >
          <AnimatePresence initial={false}>
            {transcript.map((entry, i) => (
              <motion.div
                key={i}
                initial={{ opacity: 0, y: 12 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.3 }}
                className={`flex ${entry.role === "user" ? "justify-end" : "justify-start"}`}
              >
                <div
                  className="max-w-[80%] px-4 py-3 rounded-2xl text-sm leading-relaxed"
                  style={{
                    background:
                      entry.role === "user"
                        ? "rgba(100,206,251,0.12)"
                        : "rgba(255,255,255,0.06)",
                    border:
                      entry.role === "user"
                        ? "1px solid rgba(100,206,251,0.2)"
                        : "1px solid rgba(255,255,255,0.08)",
                    color: entry.role === "user" ? "#e0f7ff" : "rgba(255,255,255,0.85)",
                  }}
                >
                  <p>{entry.text}</p>

                  {/* Citation chips */}
                  {entry.citations && entry.citations.length > 0 && (
                    <div className="flex flex-wrap gap-2 mt-3">
                      {entry.citations.map((c) => (
                        <span
                          key={c.id}
                          className="flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs"
                          style={{
                            background: "rgba(167,139,250,0.12)",
                            border: "1px solid rgba(167,139,250,0.25)",
                            color: "#c4b5fd",
                          }}
                          title={c.snippet}
                        >
                          <BookOpen size={10} />
                          {c.title}
                          <span style={{ opacity: 0.6 }}>{Math.round(c.score * 100)}%</span>
                        </span>
                      ))}
                    </div>
                  )}
                </div>
              </motion.div>
            ))}
          </AnimatePresence>

          {transcript.length === 0 && (
            <div className="flex flex-col items-center justify-center py-16 text-center">
              <Mic size={40} className="mb-4" style={{ color: "rgba(255,255,255,0.15)" }} />
              <p className="text-sm" style={{ color: "rgba(255,255,255,0.35)" }}>
                Tap the orb above to start a conversation.
              </p>
              <p className="text-xs mt-1" style={{ color: "rgba(255,255,255,0.2)" }}>
                WebSocket voice loop and RAG citations activate in Phase 9.
              </p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
