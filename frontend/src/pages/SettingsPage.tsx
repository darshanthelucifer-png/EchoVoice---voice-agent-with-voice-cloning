/**
 * SettingsPage — App & Model Settings (frontend/src/pages/SettingsPage.tsx)
 * --------------------------------------------------------------------------
 * Music-reactive theme toggle, intensity slider, manual palette override,
 * mastering preset, TTS engine picker, and other preferences.
 * Phase 8 will wire the theme settings to AudioThemeProvider.
 */

import { useState } from "react";
import { Settings, Sliders, Palette, Volume2, Music } from "lucide-react";

export function SettingsPage() {
  const [themeReactive, setThemeReactive] = useState(true);
  const [themeIntensity, setThemeIntensity] = useState(70);
  const [masteringPreset, setMasteringPreset] = useState("youtube_voiceover");
  const [ttsEngine, setTtsEngine] = useState("xtts_v2");
  const [accentManual, setAccentManual] = useState("#64CEFB");

  const save = () => {
    // Phase 8 will persist these to localStorage / backend settings
    localStorage.setItem("ev_theme_reactive", String(themeReactive));
    localStorage.setItem("ev_theme_intensity", String(themeIntensity));
    localStorage.setItem("ev_mastering_preset", masteringPreset);
    localStorage.setItem("ev_tts_engine", ttsEngine);
    localStorage.setItem("ev_accent_manual", accentManual);
    alert("Settings saved!");
  };

  return (
    <div
      className="min-h-screen pt-16"
      style={{ background: "linear-gradient(180deg, #000 0%, #050818 100%)" }}
    >
      <div className="max-w-2xl mx-auto px-4 sm:px-6 py-10 space-y-8">
        <div>
          <h1 className="text-2xl sm:text-3xl font-semibold text-white tracking-tight flex items-center gap-3">
            <Settings size={24} /> Settings
          </h1>
          <p className="text-sm mt-1 text-white/50">Configure models, theme, and audio preferences.</p>
        </div>

        {/* Theme settings */}
        <div className="glass-card p-5 space-y-5">
          <h2 className="text-sm font-semibold text-white flex items-center gap-2">
            <Palette size={14} /> Music-Reactive Theme
          </h2>

          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-white/80">Reactive theme</p>
              <p className="text-xs text-white/40">Theme changes with music energy, mood, and BPM.</p>
            </div>
            <button
              onClick={() => setThemeReactive((v) => !v)}
              className="relative w-12 h-6 rounded-full transition-colors duration-200 flex-shrink-0"
              style={{ background: themeReactive ? "rgba(100,206,251,0.5)" : "rgba(255,255,255,0.15)" }}
              aria-pressed={themeReactive}
              id="theme-reactive-toggle"
              aria-label="Toggle music-reactive theme"
            >
              <span
                className="absolute top-0.5 w-5 h-5 rounded-full bg-white transition-transform duration-200"
                style={{ transform: themeReactive ? "translateX(26px)" : "translateX(2px)" }}
              />
            </button>
          </div>

          <div className="space-y-2">
            <div className="flex justify-between">
              <label className="text-sm text-white/70" htmlFor="theme-intensity">Theme Intensity</label>
              <span className="text-xs font-mono text-white/50">{themeIntensity}%</span>
            </div>
            <input
              id="theme-intensity"
              type="range"
              min={0}
              max={100}
              value={themeIntensity}
              onChange={(e) => setThemeIntensity(Number(e.target.value))}
              className="w-full accent-cyan-400"
              disabled={!themeReactive}
            />
          </div>

          <div className="space-y-2">
            <label className="text-sm text-white/70" htmlFor="accent-color">Manual Accent Color</label>
            <div className="flex items-center gap-3">
              <input
                id="accent-color"
                type="color"
                value={accentManual}
                onChange={(e) => setAccentManual(e.target.value)}
                className="w-10 h-10 rounded-lg cursor-pointer border-0 bg-transparent"
                disabled={themeReactive}
                aria-label="Manual accent color"
              />
              <span className="text-sm font-mono text-white/60">{accentManual}</span>
              <span className="text-xs text-white/30">(active only when reactive theme is off)</span>
            </div>
          </div>
        </div>

        {/* Audio settings */}
        <div className="glass-card p-5 space-y-5">
          <h2 className="text-sm font-semibold text-white flex items-center gap-2">
            <Volume2 size={14} /> Audio & Mastering
          </h2>

          <div className="space-y-2">
            <label className="text-sm text-white/70" htmlFor="mastering-preset">Mastering Preset</label>
            <select
              id="mastering-preset"
              value={masteringPreset}
              onChange={(e) => setMasteringPreset(e.target.value)}
              className="w-full px-4 py-2.5 rounded-xl text-sm text-white outline-none"
              style={{ background: "rgba(255,255,255,0.05)", border: "1px solid rgba(255,255,255,0.1)" }}
            >
              <option value="youtube_voiceover">YouTube Voiceover (−14 LUFS)</option>
              <option value="podcast_warm">Podcast Warm (−16 LUFS)</option>
              <option value="deep_narrator">Deep Narrator (−14 LUFS, bass-boosted)</option>
              <option value="clean_neutral">Clean Neutral (−18 LUFS)</option>
            </select>
          </div>

          <div className="space-y-2">
            <label className="text-sm text-white/70" htmlFor="tts-engine">TTS Engine</label>
            <select
              id="tts-engine"
              value={ttsEngine}
              onChange={(e) => setTtsEngine(e.target.value)}
              className="w-full px-4 py-2.5 rounded-xl text-sm text-white outline-none"
              style={{ background: "rgba(255,255,255,0.05)", border: "1px solid rgba(255,255,255,0.1)" }}
            >
              <option value="xtts_v2">XTTS-v2 (Best quality)</option>
              <option value="f5_tts">F5-TTS (Fast)</option>
              <option value="openvoice">OpenVoice v2</option>
            </select>
          </div>
        </div>

        {/* Song Studio settings */}
        <div className="glass-card p-5 space-y-4">
          <h2 className="text-sm font-semibold text-white flex items-center gap-2">
            <Music size={14} /> Song Studio Defaults
          </h2>
          <p className="text-sm text-white/50">
            Per-song settings (pitch shift, sidechain, voice profile) are configured per-session in the Song Studio page.
            Global defaults can be set here in a future update.
          </p>
        </div>

        {/* Save */}
        <button
          onClick={save}
          className="w-full py-3.5 rounded-xl text-sm font-semibold text-white transition-all"
          style={{
            background: "linear-gradient(135deg, rgba(100,206,251,0.18), rgba(167,139,250,0.18))",
            border: "1px solid rgba(100,206,251,0.3)",
            minHeight: 52,
          }}
          id="settings-save-btn"
        >
          Save Settings
        </button>
      </div>
    </div>
  );
}
