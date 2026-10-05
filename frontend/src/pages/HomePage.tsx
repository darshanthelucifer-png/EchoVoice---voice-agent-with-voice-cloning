/**
 * HomePage — EchoVoice Landing Hero (frontend/src/pages/HomePage.tsx)
 * -------------------------------------------------------------------
 * Full-screen video background (CloudFront), two-column top text,
 * hero heading with ShinyText "Your Voice." line, and CTA button.
 * All built per the exact Phase 7 spec.
 */

import { useNavigate } from "react-router-dom";
import { ArrowRight } from "lucide-react";
import { ShinyText } from "../components/ui/ShinyText";

const HERO_VIDEO_URL =
  "https://d8j0ntlcm91z4.cloudfront.net/user_38xzZboKViGWJOttwIXH07lWA1P/hf_20260328_105406_16f4600d-7a92-4292-b96e-b19156c7830a.mp4";

export function HomePage() {
  const navigate = useNavigate();

  return (
    <section
      className="relative h-screen overflow-hidden"
      style={{ background: "#000000" }}
    >
      {/* ── Background Video ──────────────────────────────────────── */}
      <video
        className="absolute inset-0 w-full h-full object-cover"
        src={HERO_VIDEO_URL}
        autoPlay
        loop
        muted
        playsInline
        aria-hidden="true"
        style={{ opacity: 0.85 }}
      />

      {/* ── Dark gradient overlay to keep text legible ───────────── */}
      <div
        className="absolute inset-0"
        style={{
          background:
            "linear-gradient(to bottom, rgba(0,0,0,0.52) 0%, rgba(0,0,0,0.28) 45%, rgba(0,0,0,0.65) 100%)",
        }}
        aria-hidden="true"
      />

      {/* ── Content ───────────────────────────────────────────────── */}
      <div className="relative z-10 flex flex-col h-full pt-16">
        {/* Spacer below navbar (64px) */}
        <div className="flex-1 flex flex-col justify-between px-4 sm:px-6 lg:px-8 pb-12 max-w-7xl mx-auto w-full">

          {/* ── Top two-column section ──────────────────────────── */}
          <div className="flex flex-col lg:flex-row lg:justify-between gap-4 lg:gap-8 pt-8 sm:pt-10">
            {/* Left column */}
            <p
              className="text-sm sm:text-base max-w-xs sm:max-w-sm lg:max-w-md leading-relaxed"
              style={{ color: "rgba(255,255,255,0.78)" }}
            >
              We deliver studio-grade voice cloning and AI music tools that let
              creators speak and sing in their own voice, in any language, with
              cutting-edge realism.
            </p>

            {/* Right column — right-aligned on lg */}
            <p
              className="text-sm sm:text-base lg:text-right max-w-xs sm:max-w-sm leading-relaxed"
              style={{ color: "rgba(255,255,255,0.78)" }}
            >
              Your Voice. Any Language. Any Song !
            </p>
          </div>

          {/* ── Hero Centre ─────────────────────────────────────── */}
          <div className="flex flex-col items-start gap-4 sm:gap-6">
            {/* Eyebrow */}
            <p
              className="text-xs sm:text-sm uppercase tracking-tight"
              style={{ color: "rgba(255,255,255,0.72)" }}
              aria-label="Tagline"
            >
              Clone Your Voice In Minutes
            </p>

            {/* Main heading */}
            <h1
              className="font-medium leading-[0.85] tracking-tighter text-5xl sm:text-7xl md:text-8xl xl:text-9xl"
              style={{ lineHeight: 0.85, letterSpacing: "-0.04em" }}
            >
              {/* Line 1 — plain white */}
              <span className="block text-white">Become</span>

              {/* Line 2 — ShinyText */}
              <ShinyText
                text="Your Voice."
                baseColor="#64CEFB"
                shineColor="#ffffff"
                speed={3}
                spread={100}
                className="block"
              />
            </h1>

            {/* ── CTA Button ──────────────────────────────────── */}
            <button
              id="hero-cta-voice-studio"
              onClick={() => navigate("/voice-studio")}
              className="group flex items-center gap-3 mt-2 px-6 md:px-8 py-3 md:py-4 rounded-full text-white text-sm sm:text-base font-medium transition-all duration-300 hover:bg-gray-900 focus-visible:ring-2 focus-visible:ring-white/50"
              style={{
                background: "#000000",
                border: "1px solid rgba(255,255,255,0.18)",
                minHeight: 52,
              }}
              aria-label="Start Voice Studio"
            >
              Start Voice Studio
              <ArrowRight
                size={18}
                className="transition-transform duration-300 group-hover:translate-x-1"
                aria-hidden="true"
              />
            </button>
          </div>

        </div>
      </div>
    </section>
  );
}
