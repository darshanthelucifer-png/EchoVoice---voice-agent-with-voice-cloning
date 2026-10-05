/**
 * KnowledgePage — RAG Knowledge Base (frontend/src/pages/KnowledgePage.tsx)
 * --------------------------------------------------------------------------
 * Document ingestion (PDF, DOCX, TXT), FAISS vector store, BGE-M3 embeddings,
 * BM25 hybrid search, and confidence gating — all existing Phase 3/4 functionality
 * restyled with glass cards as required.
 */

import { useState, useRef } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Upload, BookOpen, Search, FileText, X, Loader2, AlertTriangle, CheckCircle } from "lucide-react";
import { API_BASE, getAuthToken } from "../services/api";

interface KBDocument {
  id: string;
  filename: string;
  chunks: number;
  created_at: string;
}

interface SearchResult {
  text: string;
  score: number;
  source: string;
}

export function KnowledgePage() {
  const [docs, setDocs] = useState<KBDocument[]>([]);
  const [uploading, setUploading] = useState(false);
  const [uploadError, setUploadError] = useState<string | null>(null);
  const [uploadSuccess, setUploadSuccess] = useState<string | null>(null);
  const [query, setQuery] = useState("");
  const [searching, setSearching] = useState(false);
  const [searchResults, setSearchResults] = useState<SearchResult[]>([]);
  const fileRef = useRef<HTMLInputElement>(null);

  const handleUpload = async (file: File) => {
    setUploading(true);
    setUploadError(null);
    setUploadSuccess(null);
    try {
      const token = getAuthToken();
      const fd = new FormData();
      fd.append("file", file);
      const res = await fetch(`${API_BASE}/rag/upload`, {
        method: "POST",
        headers: token ? { Authorization: `Bearer ${token}` } : {},
        body: fd,
      });
      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        throw new Error(err.detail || "Upload failed");
      }
      const { data } = await res.json();
      setUploadSuccess(`Indexed "${file.name}" → ${data.chunks_created || "?"} chunks`);
      setDocs((prev) => [
        { id: data.document_id, filename: file.name, chunks: data.chunks_created, created_at: new Date().toISOString() },
        ...prev,
      ]);
    } catch (e: unknown) {
      setUploadError(e instanceof Error ? e.message : "Upload failed");
    } finally {
      setUploading(false);
    }
  };

  const handleSearch = async () => {
    if (!query.trim()) return;
    setSearching(true);
    setSearchResults([]);
    try {
      const token = getAuthToken();
      const res = await fetch(`${API_BASE}/rag/query`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          ...(token ? { Authorization: `Bearer ${token}` } : {}),
        },
        body: JSON.stringify({ query, top_k: 5 }),
      });
      if (!res.ok) throw new Error("Search failed");
      const { data } = await res.json();
      setSearchResults(data.results || []);
    } catch {
      setSearchResults([]);
    } finally {
      setSearching(false);
    }
  };

  return (
    <div
      className="min-h-screen pt-16"
      style={{ background: "linear-gradient(180deg, #000 0%, #050818 100%)" }}
    >
      <div className="max-w-3xl mx-auto px-4 sm:px-6 py-10 space-y-8">
        <div>
          <h1 className="text-2xl sm:text-3xl font-semibold text-white tracking-tight">Knowledge Base</h1>
          <p className="text-sm mt-1 text-white/50">
            Upload documents (PDF, DOCX, TXT) → power your voice agent with RAG.
          </p>
        </div>

        {/* Upload */}
        <div
          className="glass-card p-6 cursor-pointer hover:border-white/20 transition-colors"
          onClick={() => fileRef.current?.click()}
          role="button"
          tabIndex={0}
          onKeyDown={(e) => e.key === "Enter" && fileRef.current?.click()}
          aria-label="Upload document"
        >
          <input
            ref={fileRef}
            type="file"
            accept=".pdf,.docx,.txt,.md"
            className="hidden"
            onChange={(e) => e.target.files?.[0] && handleUpload(e.target.files[0])}
          />
          <div className="flex flex-col items-center gap-3 py-4">
            {uploading ? <Loader2 size={32} className="animate-spin text-cyan-400" /> : <Upload size={32} style={{ color: "rgba(100,206,251,0.7)" }} />}
            <p className="text-sm text-white/70">Drop document here or click to browse</p>
            <p className="text-xs text-white/30">PDF, DOCX, TXT, Markdown</p>
          </div>
        </div>

        {uploadSuccess && (
          <div className="glass-card p-4 flex items-center gap-3" style={{ borderColor: "rgba(52,211,153,0.3)" }}>
            <CheckCircle size={16} className="text-green-400" />
            <p className="text-sm text-green-300">{uploadSuccess}</p>
          </div>
        )}
        {uploadError && (
          <div className="glass-card p-4 flex items-center gap-3" style={{ borderColor: "rgba(239,68,68,0.3)" }}>
            <AlertTriangle size={16} className="text-red-400" />
            <p className="text-sm text-red-300">{uploadError}</p>
          </div>
        )}

        {/* Search */}
        <div className="glass-card p-5 space-y-4">
          <h2 className="text-sm font-semibold text-white flex items-center gap-2"><Search size={14} /> Hybrid Search</h2>
          <div className="flex gap-3">
            <input
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && handleSearch()}
              placeholder="Ask anything about your documents…"
              className="flex-1 px-4 py-2.5 rounded-xl text-sm text-white placeholder:text-white/25 outline-none"
              style={{ background: "rgba(255,255,255,0.05)", border: "1px solid rgba(255,255,255,0.1)" }}
              id="knowledge-search-input"
            />
            <button
              onClick={handleSearch}
              disabled={!query.trim() || searching}
              className="px-5 py-2.5 rounded-xl text-sm font-medium text-white transition-all disabled:opacity-40"
              style={{ background: "rgba(100,206,251,0.15)", border: "1px solid rgba(100,206,251,0.3)", minHeight: 44 }}
              id="knowledge-search-btn"
            >
              {searching ? <Loader2 size={14} className="animate-spin" /> : "Search"}
            </button>
          </div>

          <AnimatePresence>
            {searchResults.length > 0 && (
              <motion.div className="space-y-3" initial={{ opacity: 0 }} animate={{ opacity: 1 }}>
                {searchResults.map((r, i) => (
                  <div key={i} className="p-4 rounded-xl" style={{ background: "rgba(255,255,255,0.03)", border: "1px solid rgba(255,255,255,0.07)" }}>
                    <div className="flex items-center justify-between mb-2">
                      <span className="text-xs text-white/40 flex items-center gap-1"><FileText size={10} /> {r.source}</span>
                      <span className="text-xs font-mono" style={{ color: r.score >= 0.7 ? "#34d399" : "#f59e0b" }}>
                        {Math.round(r.score * 100)}%
                      </span>
                    </div>
                    <p className="text-sm text-white/80 leading-relaxed">{r.text}</p>
                  </div>
                ))}
              </motion.div>
            )}
          </AnimatePresence>
        </div>

        {/* Document library */}
        {docs.length > 0 && (
          <div className="glass-card p-5 space-y-3">
            <h2 className="text-sm font-semibold text-white flex items-center gap-2"><BookOpen size={14} /> Indexed Documents</h2>
            {docs.map((d) => (
              <div key={d.id} className="flex items-center justify-between p-3 rounded-xl" style={{ background: "rgba(255,255,255,0.03)", border: "1px solid rgba(255,255,255,0.06)" }}>
                <div className="flex items-center gap-3">
                  <FileText size={14} style={{ color: "rgba(100,206,251,0.7)" }} />
                  <div>
                    <p className="text-sm text-white/80">{d.filename}</p>
                    <p className="text-xs text-white/30">{d.chunks} chunks indexed</p>
                  </div>
                </div>
                <button
                  onClick={() => setDocs((prev) => prev.filter((x) => x.id !== d.id))}
                  className="p-1.5 rounded-full text-white/30 hover:text-white/70 transition-colors"
                  aria-label={`Remove ${d.filename}`}
                >
                  <X size={12} />
                </button>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
