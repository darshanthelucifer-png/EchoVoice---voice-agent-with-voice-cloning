/**
 * DashboardPage — Metrics, Latency & Quality Dashboard (frontend/src/pages/DashboardPage.tsx)
 * --------------------------------------------------------------------------------------------
 * Displays real-time system metrics: model latency, similarity scores, job history,
 * audio quality stats. Glass cards restyled per Phase 7 spec.
 */

import { useState, useEffect } from "react";
import { motion } from "framer-motion";
import { Activity, Clock, Cpu, Zap, BarChart2, CheckCircle } from "lucide-react";
import { API_BASE, getAuthToken } from "../services/api";

interface HealthMetrics {
  status: string;
  models_loaded: Record<string, boolean>;
  average_tts_latency_ms?: number;
  average_similarity_score?: number;
  jobs_completed?: number;
  jobs_failed?: number;
  uptime_seconds?: number;
}

function MetricCard({ label, value, unit, icon: Icon, color }: {
  label: string; value: string | number; unit?: string; icon: React.ComponentType<{size?: number; color?: string}>; color: string;
}) {
  return (
    <motion.div
      className="glass-card p-5 flex flex-col gap-3"
      initial={{ opacity: 0, y: 12 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.4 }}
    >
      <div className="flex items-center justify-between">
        <p className="text-xs text-white/50">{label}</p>
        <Icon size={16} color={color} />
      </div>
      <div className="flex items-baseline gap-1">
        <span className="text-2xl font-semibold text-white">{value}</span>
        {unit && <span className="text-sm text-white/40">{unit}</span>}
      </div>
    </motion.div>
  );
}

export function DashboardPage() {
  const [metrics, setMetrics] = useState<HealthMetrics | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchMetrics = async () => {
      try {
        const token = getAuthToken();
        const res = await fetch(`${API_BASE}/health`, {
          headers: token ? { Authorization: `Bearer ${token}` } : {},
        });
        if (res.ok) {
          const { data } = await res.json();
          setMetrics(data);
        }
      } catch {
        // Backend not available
      } finally {
        setLoading(false);
      }
    };
    fetchMetrics();
    const interval = setInterval(fetchMetrics, 30_000);
    return () => clearInterval(interval);
  }, []);

  return (
    <div
      className="min-h-screen pt-16"
      style={{ background: "linear-gradient(180deg, #000 0%, #050818 100%)" }}
    >
      <div className="max-w-5xl mx-auto px-4 sm:px-6 py-10 space-y-8">
        <div>
          <h1 className="text-2xl sm:text-3xl font-semibold text-white tracking-tight">Dashboard</h1>
          <p className="text-sm mt-1 text-white/50">System metrics, model latency, and quality scores.</p>
        </div>

        {/* Status pill */}
        <div className="flex items-center gap-2">
          <div
            className="w-2 h-2 rounded-full"
            style={{ background: metrics?.status === "ok" ? "#34d399" : "#f87171" }}
          />
          <span className="text-sm text-white/70">
            {loading ? "Connecting to backend…" : metrics?.status === "ok" ? "All systems operational" : "Backend unavailable — start the FastAPI server"}
          </span>
        </div>

        {/* Metric cards grid */}
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
          <MetricCard
            label="TTS Latency"
            value={metrics?.average_tts_latency_ms != null ? Math.round(metrics.average_tts_latency_ms) : "—"}
            unit="ms"
            icon={Clock}
            color="#64CEFB"
          />
          <MetricCard
            label="Voice Match"
            value={metrics?.average_similarity_score != null ? `${Math.round(metrics.average_similarity_score * 100)}%` : "—"}
            icon={Activity}
            color="#34d399"
          />
          <MetricCard
            label="Jobs Done"
            value={metrics?.jobs_completed ?? "—"}
            icon={CheckCircle}
            color="#a78bfa"
          />
          <MetricCard
            label="Uptime"
            value={metrics?.uptime_seconds != null ? `${Math.floor(metrics.uptime_seconds / 60)}m` : "—"}
            icon={Zap}
            color="#f59e0b"
          />
        </div>

        {/* Model status table */}
        {metrics?.models_loaded && (
          <div className="glass-card p-5 space-y-3">
            <h2 className="text-sm font-semibold text-white flex items-center gap-2">
              <Cpu size={14} /> Loaded Models
            </h2>
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
              {Object.entries(metrics.models_loaded).map(([model, loaded]) => (
                <div
                  key={model}
                  className="flex items-center justify-between px-4 py-3 rounded-xl"
                  style={{ background: "rgba(255,255,255,0.03)", border: "1px solid rgba(255,255,255,0.06)" }}
                >
                  <span className="text-sm text-white/70">{model}</span>
                  <span
                    className="text-xs px-2 py-0.5 rounded-full"
                    style={{
                      background: loaded ? "rgba(52,211,153,0.12)" : "rgba(239,68,68,0.12)",
                      color: loaded ? "#34d399" : "#f87171",
                    }}
                  >
                    {loaded ? "Ready" : "Not loaded"}
                  </span>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Placeholder latency chart */}
        <div className="glass-card p-5 space-y-3">
          <h2 className="text-sm font-semibold text-white flex items-center gap-2">
            <BarChart2 size={14} /> Real-Time Latency Trend
          </h2>
          <div
            className="h-40 rounded-xl flex items-center justify-center"
            style={{ background: "rgba(255,255,255,0.02)", border: "1px dashed rgba(255,255,255,0.07)" }}
          >
            <p className="text-sm text-white/25">Live latency chart activates in Phase 9</p>
          </div>
        </div>
      </div>
    </div>
  );
}
