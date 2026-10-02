/**
 * EchoVoice Main App Shell (frontend/src/App.tsx)
 * ------------------------------------------------
 */

import React, { useState, useEffect } from "react";
import {
  Mic,
  Sparkles,
  Radio,
  BookOpen,
  User,
  LogIn,
  LogOut,
} from "lucide-react";
import { VoiceStudio } from "./components/VoiceStudio";
import { AuthModal } from "./components/AuthModal";
import {
  fetchCurrentUser,
  ensureAuthenticatedSession,
  clearAuthToken,
} from "./services/api";
import type { UserProfile } from "./services/api";

export const App: React.FC = () => {
  const [activeTab, setActiveTab] = useState<"studio" | "agent" | "knowledge">("studio");
  const [currentUser, setCurrentUser] = useState<UserProfile | null>(null);
  const [isAuthModalOpen, setIsAuthModalOpen] = useState<boolean>(false);

  // Auto-bootstrap active session on mount
  useEffect(() => {
    const initAuth = async () => {
      try {
        await ensureAuthenticatedSession();
        const user = await fetchCurrentUser();
        if (user) {
          setCurrentUser(user);
        }
      } catch (err) {
        console.warn("Session auto-bootstrap notice:", err);
      }
    };
    initAuth();
  }, []);

  const handleSignOut = () => {
    clearAuthToken();
    setCurrentUser(null);
  };

  return (
    <div style={{ minHeight: "100vh", display: "flex", flexDirection: "column" }}>
      {/* Top Navbar */}
      <header
        style={{
          borderBottom: "1px solid rgba(255, 255, 255, 0.08)",
          background: "rgba(9, 13, 22, 0.85)",
          backdropFilter: "blur(12px)",
          position: "sticky",
          top: 0,
          zIndex: 50,
        }}
      >
        <div
          style={{
            maxWidth: "1280px",
            margin: "0 auto",
            padding: "0.85rem 1.5rem",
            display: "flex",
            alignItems: "center",
            justifyContent: "space-between",
          }}
        >
          {/* Brand Logo */}
          <div style={{ display: "flex", alignItems: "center", gap: "0.85rem" }}>
            <div
              style={{
                width: "38px",
                height: "38px",
                borderRadius: "10px",
                background: "linear-gradient(135deg, #6366f1 0%, #06b6d4 100%)",
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
                boxShadow: "0 0 15px rgba(99, 102, 241, 0.5)",
              }}
            >
              <Mic size={22} color="#ffffff" />
            </div>
            <div>
              <div style={{ display: "flex", alignItems: "center", gap: "0.5rem" }}>
                <span className="font-heading" style={{ fontSize: "1.25rem", fontWeight: 700, letterSpacing: "-0.02em" }}>
                  EchoVoice
                </span>
                <span
                  style={{
                    background: "rgba(99, 102, 241, 0.2)",
                    color: "#a5b4fc",
                    fontSize: "0.65rem",
                    fontWeight: 700,
                    padding: "0.15rem 0.45rem",
                    borderRadius: "4px",
                    textTransform: "uppercase",
                  }}
                >
                  v1.0 Free
                </span>
              </div>
              <div style={{ fontSize: "0.7rem", color: "#64748b" }}>
                Conversational AI Agent & Studio Voice Cloning
              </div>
            </div>
          </div>

          {/* Navigation Tabs */}
          <nav style={{ display: "flex", gap: "0.5rem" }}>
            <button
              onClick={() => setActiveTab("studio")}
              style={{
                background: activeTab === "studio" ? "rgba(99, 102, 241, 0.2)" : "transparent",
                color: activeTab === "studio" ? "#c7d2fe" : "#94a3b8",
                border: activeTab === "studio" ? "1px solid rgba(99, 102, 241, 0.4)" : "1px solid transparent",
                padding: "0.5rem 1rem",
                borderRadius: "0.5rem",
                fontSize: "0.85rem",
                fontWeight: 600,
                cursor: "pointer",
                display: "flex",
                alignItems: "center",
                gap: "0.4rem",
              }}
            >
              <Sparkles size={16} /> Voice Studio (Phase 4–6)
            </button>

            <button
              onClick={() => setActiveTab("agent")}
              style={{
                background: activeTab === "agent" ? "rgba(99, 102, 241, 0.2)" : "transparent",
                color: activeTab === "agent" ? "#c7d2fe" : "#64748b",
                border: "1px solid transparent",
                padding: "0.5rem 1rem",
                borderRadius: "0.5rem",
                fontSize: "0.85rem",
                fontWeight: 600,
                cursor: "pointer",
                display: "flex",
                alignItems: "center",
                gap: "0.4rem",
              }}
            >
              <Radio size={16} /> Real-Time Agent (Phase 9)
            </button>

            <button
              onClick={() => setActiveTab("knowledge")}
              style={{
                background: activeTab === "knowledge" ? "rgba(99, 102, 241, 0.2)" : "transparent",
                color: activeTab === "knowledge" ? "#c7d2fe" : "#64748b",
                border: "1px solid transparent",
                padding: "0.5rem 1rem",
                borderRadius: "0.5rem",
                fontSize: "0.85rem",
                fontWeight: 600,
                cursor: "pointer",
                display: "flex",
                alignItems: "center",
                gap: "0.4rem",
              }}
            >
              <BookOpen size={16} /> Knowledge RAG (Phase 8)
            </button>
          </nav>

          {/* User & Model Status Pill */}
          <div style={{ display: "flex", alignItems: "center", gap: "0.75rem" }}>
            {/* User Account Controls */}
            {currentUser ? (
              <div style={{ display: "flex", alignItems: "center", gap: "0.4rem" }}>
                <button
                  onClick={() => setIsAuthModalOpen(true)}
                  style={{
                    background: "rgba(99, 102, 241, 0.15)",
                    border: "1px solid rgba(99, 102, 241, 0.35)",
                    color: "#c7d2fe",
                    fontSize: "0.75rem",
                    fontWeight: 600,
                    padding: "0.35rem 0.75rem",
                    borderRadius: "9999px",
                    display: "flex",
                    alignItems: "center",
                    gap: "0.45rem",
                    cursor: "pointer",
                  }}
                  title="Switch user account"
                >
                  <User size={13} color="#818cf8" />
                  <span>{currentUser.full_name || currentUser.email}</span>
                </button>
                <button
                  onClick={handleSignOut}
                  style={{
                    background: "rgba(255, 255, 255, 0.05)",
                    border: "1px solid rgba(255, 255, 255, 0.1)",
                    color: "#94a3b8",
                    padding: "0.35rem 0.5rem",
                    borderRadius: "9999px",
                    cursor: "pointer",
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "center",
                  }}
                  title="Sign Out"
                >
                  <LogOut size={13} />
                </button>
              </div>
            ) : (
              <button
                onClick={() => setIsAuthModalOpen(true)}
                className="gradient-btn"
                style={{
                  fontSize: "0.75rem",
                  fontWeight: 600,
                  padding: "0.35rem 0.85rem",
                  borderRadius: "9999px",
                  display: "flex",
                  alignItems: "center",
                  gap: "0.4rem",
                  cursor: "pointer",
                }}
              >
                <LogIn size={13} />
                Sign In / Demo
              </button>
            )}

            {/* Model Status Pill */}
            <div
              style={{
                background: "rgba(16, 185, 129, 0.15)",
                border: "1px solid rgba(16, 185, 129, 0.3)",
                color: "#34d399",
                fontSize: "0.75rem",
                fontWeight: 600,
                padding: "0.35rem 0.75rem",
                borderRadius: "9999px",
                display: "flex",
                alignItems: "center",
                gap: "0.4rem",
              }}
            >
              <span style={{ width: "8px", height: "8px", borderRadius: "50%", background: "#10b981", display: "inline-block" }} />
              XTTS-v2 / Connected
            </div>
          </div>
        </div>
      </header>

      {/* Main Content Area */}
      <main style={{ flex: 1 }}>
        {activeTab === "studio" && (
          <VoiceStudio
            currentUser={currentUser}
            onOpenAuth={() => setIsAuthModalOpen(true)}
          />
        )}

        {activeTab === "agent" && (
          <div style={{ maxWidth: "800px", margin: "4rem auto", textAlign: "center", padding: "2rem" }} className="glass-card">
            <Radio size={48} color="#818cf8" style={{ marginBottom: "1rem" }} />
            <h2 style={{ fontSize: "1.5rem", fontWeight: 700, marginBottom: "0.5rem" }}>
              Real-Time Conversational Voice Agent
            </h2>
            <p style={{ color: "#94a3b8", maxWidth: "540px", margin: "0 auto 1.5rem" }}>
              The full duplex voice loop with Silero VAD, streaming Whisper ASR, RAG citations, and barge-in interruption will be activated in Phases 7–9.
            </p>
            <button onClick={() => setActiveTab("studio")} className="gradient-btn" style={{ padding: "0.75rem 1.5rem", borderRadius: "0.5rem", cursor: "pointer" }}>
              Return to Voice Studio
            </button>
          </div>
        )}

        {activeTab === "knowledge" && (
          <div style={{ maxWidth: "800px", margin: "4rem auto", textAlign: "center", padding: "2rem" }} className="glass-card">
            <BookOpen size={48} color="#06b6d4" style={{ marginBottom: "1rem" }} />
            <h2 style={{ fontSize: "1.5rem", fontWeight: 700, marginBottom: "0.5rem" }}>
              RAG Knowledge Base
            </h2>
            <p style={{ color: "#94a3b8", maxWidth: "540px", margin: "0 auto 1.5rem" }}>
              Document ingestion (PDF, DOCX, TXT), FAISS vector store, BGE-M3 embeddings, BM25 hybrid search, and confidence gating will be activated in Phase 8.
            </p>
            <button onClick={() => setActiveTab("studio")} className="gradient-btn" style={{ padding: "0.75rem 1.5rem", borderRadius: "0.5rem", cursor: "pointer" }}>
              Return to Voice Studio
            </button>
          </div>
        )}
      </main>

      {/* Auth Modal */}
      <AuthModal
        isOpen={isAuthModalOpen}
        onClose={() => setIsAuthModalOpen(false)}
        onAuthSuccess={(user) => {
          setCurrentUser(user);
        }}
      />

      {/* Footer */}
      <footer
        style={{
          borderTop: "1px solid rgba(255, 255, 255, 0.06)",
          padding: "1.25rem 2rem",
          fontSize: "0.8rem",
          color: "#64748b",
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
          maxWidth: "1280px",
          margin: "0 auto",
          width: "100%",
        }}
      >
        <div>EchoVoice — 100% Free Open-Source Voice Architecture</div>
        <div style={{ display: "flex", gap: "1rem" }}>
          <span>YouTube -14 LUFS Standard</span>
          <span>•</span>
          <span>Zero-Shot Voice Cloning</span>
          <span>•</span>
          <span>Inaudible AI Watermarked</span>
        </div>
      </footer>
    </div>
  );
};

export default App;
