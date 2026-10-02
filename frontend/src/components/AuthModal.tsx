import React, { useState } from "react";
import { User, Lock, Mail, Sparkles, X, CheckCircle2, ShieldCheck, AlertCircle } from "lucide-react";
import { loginUser, registerUser, ensureAuthenticatedSession, fetchCurrentUser } from "../services/api";
import type { UserProfile } from "../services/api";

interface AuthModalProps {
  isOpen: boolean;
  onClose: () => void;
  onAuthSuccess: (user: UserProfile) => void;
}

export const AuthModal: React.FC<AuthModalProps> = ({ isOpen, onClose, onAuthSuccess }) => {
  const [mode, setMode] = useState<"login" | "register">("login");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [fullName, setFullName] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  if (!isOpen) return null;

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError(null);
    setLoading(true);

    try {
      if (mode === "login") {
        await loginUser(email.trim(), password);
      } else {
        if (password.length < 8) {
          throw new Error("Password must be at least 8 characters long.");
        }
        await registerUser(email.trim(), password, fullName.trim() || undefined);
      }
      const user = await fetchCurrentUser();
      if (user) {
        onAuthSuccess(user);
        onClose();
      } else {
        throw new Error("Failed to load user profile after authentication.");
      }
    } catch (err: any) {
      setError(err.message || "Authentication failed");
    } finally {
      setLoading(false);
    }
  };

  const handleQuickDemo = async () => {
    setError(null);
    setLoading(true);
    try {
      await ensureAuthenticatedSession();
      const user = await fetchCurrentUser();
      if (user) {
        onAuthSuccess(user);
        onClose();
      }
    } catch (err: any) {
      setError(err.message || "Failed to start demo session");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div
      style={{
        position: "fixed",
        inset: 0,
        zIndex: 100,
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        background: "rgba(3, 7, 18, 0.8)",
        backdropFilter: "blur(12px)",
        padding: "1rem",
      }}
    >
      <div
        className="glass-card"
        style={{
          width: "100%",
          maxWidth: "460px",
          padding: "2rem",
          borderRadius: "1rem",
          position: "relative",
          boxShadow: "0 25px 50px -12px rgba(0, 0, 0, 0.7), 0 0 30px rgba(99, 102, 241, 0.2)",
        }}
      >
        {/* Close Button */}
        <button
          onClick={onClose}
          style={{
            position: "absolute",
            top: "1.25rem",
            right: "1.25rem",
            background: "transparent",
            border: "none",
            color: "#94a3b8",
            cursor: "pointer",
          }}
        >
          <X size={20} />
        </button>

        {/* Modal Header */}
        <div style={{ textAlign: "center", marginBottom: "1.5rem" }}>
          <div
            style={{
              width: "48px",
              height: "48px",
              borderRadius: "12px",
              background: "linear-gradient(135deg, #6366f1 0%, #06b6d4 100%)",
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              margin: "0 auto 0.75rem",
              boxShadow: "0 0 15px rgba(99, 102, 241, 0.5)",
            }}
          >
            <ShieldCheck size={26} color="#fff" />
          </div>
          <h2 style={{ fontSize: "1.35rem", fontWeight: 700, color: "#f8fafc" }}>
            {mode === "login" ? "Welcome Back to EchoVoice" : "Create EchoVoice Account"}
          </h2>
          <p style={{ fontSize: "0.85rem", color: "#94a3b8", marginTop: "0.35rem" }}>
            {mode === "login"
              ? "Sign in to access your cloned voices and speech synthesis studio"
              : "Register to manage ethical voice profiles and secure voice models"}
          </p>
        </div>

        {/* Instant Demo Shortcut */}
        <button
          onClick={handleQuickDemo}
          disabled={loading}
          style={{
            width: "100%",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            gap: "0.5rem",
            background: "rgba(99, 102, 241, 0.15)",
            border: "1px solid rgba(99, 102, 241, 0.4)",
            color: "#c7d2fe",
            padding: "0.75rem 1rem",
            borderRadius: "0.6rem",
            fontWeight: 600,
            fontSize: "0.85rem",
            cursor: "pointer",
            marginBottom: "1.25rem",
            transition: "all 0.2s ease",
          }}
        >
          <Sparkles size={16} color="#a5b4fc" />
          ⚡ 1-Click Instant Demo Login (demo@echovoice.ai)
        </button>

        <div style={{ display: "flex", alignItems: "center", gap: "0.75rem", marginBottom: "1.25rem" }}>
          <div style={{ flex: 1, height: "1px", background: "rgba(255, 255, 255, 0.1)" }} />
          <span style={{ fontSize: "0.75rem", color: "#64748b", textTransform: "uppercase" }}>or sign in manually</span>
          <div style={{ flex: 1, height: "1px", background: "rgba(255, 255, 255, 0.1)" }} />
        </div>

        {/* Tabs */}
        <div
          style={{
            display: "flex",
            background: "rgba(2, 6, 23, 0.6)",
            padding: "0.25rem",
            borderRadius: "0.5rem",
            marginBottom: "1.25rem",
          }}
        >
          <button
            type="button"
            onClick={() => { setMode("login"); setError(null); }}
            style={{
              flex: 1,
              padding: "0.5rem",
              borderRadius: "0.4rem",
              fontSize: "0.85rem",
              fontWeight: 600,
              background: mode === "login" ? "rgba(99, 102, 241, 0.25)" : "transparent",
              color: mode === "login" ? "#e0e7ff" : "#94a3b8",
              border: "none",
              cursor: "pointer",
            }}
          >
            Sign In
          </button>
          <button
            type="button"
            onClick={() => { setMode("register"); setError(null); }}
            style={{
              flex: 1,
              padding: "0.5rem",
              borderRadius: "0.4rem",
              fontSize: "0.85rem",
              fontWeight: 600,
              background: mode === "register" ? "rgba(99, 102, 241, 0.25)" : "transparent",
              color: mode === "register" ? "#e0e7ff" : "#94a3b8",
              border: "none",
              cursor: "pointer",
            }}
          >
            Register
          </button>
        </div>

        {error && (
          <div
            style={{
              display: "flex",
              alignItems: "center",
              gap: "0.5rem",
              background: "rgba(244, 63, 94, 0.15)",
              border: "1px solid rgba(244, 63, 94, 0.3)",
              color: "#fda4af",
              fontSize: "0.8rem",
              padding: "0.6rem 0.85rem",
              borderRadius: "0.5rem",
              marginBottom: "1rem",
            }}
          >
            <AlertCircle size={16} />
            <span>{error}</span>
          </div>
        )}

        <form onSubmit={handleSubmit} style={{ display: "flex", flexDirection: "column", gap: "1rem" }}>
          {mode === "register" && (
            <div>
              <label style={{ display: "block", fontSize: "0.8rem", color: "#cbd5e1", marginBottom: "0.35rem" }}>
                Full Name
              </label>
              <div style={{ position: "relative" }}>
                <User size={16} color="#64748b" style={{ position: "absolute", left: "0.75rem", top: "0.75rem" }} />
                <input
                  type="text"
                  required
                  value={fullName}
                  onChange={(e) => setFullName(e.target.value)}
                  placeholder="e.g. Alex Mercer"
                  style={{
                    width: "100%",
                    background: "rgba(15, 23, 42, 0.6)",
                    border: "1px solid rgba(255, 255, 255, 0.15)",
                    borderRadius: "0.5rem",
                    padding: "0.65rem 0.75rem 0.65rem 2.25rem",
                    color: "#fff",
                    fontSize: "0.85rem",
                  }}
                />
              </div>
            </div>
          )}

          <div>
            <label style={{ display: "block", fontSize: "0.8rem", color: "#cbd5e1", marginBottom: "0.35rem" }}>
              Email Address
            </label>
            <div style={{ position: "relative" }}>
              <Mail size={16} color="#64748b" style={{ position: "absolute", left: "0.75rem", top: "0.75rem" }} />
              <input
                type="email"
                required
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                placeholder="you@domain.com"
                style={{
                  width: "100%",
                  background: "rgba(15, 23, 42, 0.6)",
                  border: "1px solid rgba(255, 255, 255, 0.15)",
                  borderRadius: "0.5rem",
                  padding: "0.65rem 0.75rem 0.65rem 2.25rem",
                  color: "#fff",
                  fontSize: "0.85rem",
                }}
              />
            </div>
          </div>

          <div>
            <label style={{ display: "block", fontSize: "0.8rem", color: "#cbd5e1", marginBottom: "0.35rem" }}>
              Password {mode === "register" && "(min 8 characters)"}
            </label>
            <div style={{ position: "relative" }}>
              <Lock size={16} color="#64748b" style={{ position: "absolute", left: "0.75rem", top: "0.75rem" }} />
              <input
                type="password"
                required
                minLength={8}
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                placeholder="••••••••"
                style={{
                  width: "100%",
                  background: "rgba(15, 23, 42, 0.6)",
                  border: "1px solid rgba(255, 255, 255, 0.15)",
                  borderRadius: "0.5rem",
                  padding: "0.65rem 0.75rem 0.65rem 2.25rem",
                  color: "#fff",
                  fontSize: "0.85rem",
                }}
              />
            </div>
          </div>

          <button
            type="submit"
            disabled={loading}
            className="gradient-btn"
            style={{
              padding: "0.75rem",
              borderRadius: "0.5rem",
              fontWeight: 700,
              fontSize: "0.9rem",
              cursor: "pointer",
              marginTop: "0.5rem",
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              gap: "0.5rem",
            }}
          >
            <CheckCircle2 size={16} />
            {loading ? "Authenticating..." : mode === "login" ? "Sign In" : "Create Account"}
          </button>
        </form>
      </div>
    </div>
  );
};
