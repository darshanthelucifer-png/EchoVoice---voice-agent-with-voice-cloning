/**
 * Navbar — EchoVoice top navigation bar (frontend/src/components/layout/Navbar.tsx)
 * -----------------------------------------------------------------------------------
 * Renders the pill-shaped navigation container per spec:
 * - Logo: circular white-border circle with inner white circle + "EchoVoice" text.
 * - Links: Home, Talk, Voice Studio, Song Studio, Knowledge, Dashboard, Contact us (with arrow).
 * - Mobile: hamburger Menu icon → slide-in drawer.
 * - Auth state: Sign In button or user avatar pill + sign-out.
 */

import { useState } from "react";
import { Link, useLocation } from "react-router-dom";
import { Menu, X, ArrowUpRight, LogIn, LogOut, User } from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";

interface NavbarProps {
  currentUser: { full_name?: string; email?: string } | null;
  onOpenAuth: () => void;
  onSignOut: () => void;
}

const NAV_LINKS = [
  { label: "Home", to: "/" },
  { label: "Talk", to: "/talk" },
  { label: "Voice Studio", to: "/voice-studio" },
  { label: "Song Studio", to: "/song-studio" },
  { label: "Knowledge", to: "/knowledge" },
  { label: "Dashboard", to: "/dashboard" },
];

export function Navbar({ currentUser, onOpenAuth, onSignOut }: NavbarProps) {
  const [drawerOpen, setDrawerOpen] = useState(false);
  const location = useLocation();

  const isActive = (to: string) =>
    to === "/" ? location.pathname === "/" : location.pathname.startsWith(to);

  return (
    <>
      <nav
        className="fixed top-0 left-0 right-0 z-50"
        style={{
          background: "rgba(0,0,0,0.55)",
          backdropFilter: "blur(20px)",
          WebkitBackdropFilter: "blur(20px)",
          borderBottom: "1px solid rgba(255,255,255,0.06)",
        }}
        aria-label="Main navigation"
      >
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between h-16">
            {/* ── Logo ─────────────────────────────────────────────── */}
            <Link
              to="/"
              className="flex items-center gap-3 select-none"
              aria-label="EchoVoice home"
            >
              {/* Circular logo: outer ring + inner filled dot */}
              <div
                className="relative flex-shrink-0"
                style={{ width: 36, height: 36 }}
              >
                {/* Outer circle — 2px white border */}
                <div
                  className="absolute inset-0 rounded-full"
                  style={{ border: "2px solid #ffffff" }}
                />
                {/* Inner filled white circle, centred */}
                <div
                  className="absolute rounded-full bg-white"
                  style={{
                    width: 16,
                    height: 16,
                    top: "50%",
                    left: "50%",
                    transform: "translate(-50%, -50%)",
                  }}
                />
              </div>
              <span
                className="text-white font-semibold text-lg tracking-tight"
                style={{ letterSpacing: "-0.02em" }}
              >
                EchoVoice
              </span>
            </Link>

            {/* ── Desktop Nav Pill ─────────────────────────────────── */}
            <div className="hidden lg:flex items-center">
              <div
                className="flex items-center gap-1 px-3 py-2 rounded-full"
                style={{
                  border: "1px solid rgb(55, 65, 81)", /* border-gray-700 */
                  background: "rgba(255,255,255,0.03)",
                }}
              >
                {NAV_LINKS.map((link) => (
                  <Link
                    key={link.to}
                    to={link.to}
                    className="px-3 py-1.5 rounded-full text-sm transition-all duration-200 whitespace-nowrap"
                    style={{
                      color: isActive(link.to)
                        ? "#ffffff"
                        : "rgba(255,255,255,0.75)",
                      background: isActive(link.to)
                        ? "rgba(255,255,255,0.1)"
                        : "transparent",
                    }}
                    onMouseEnter={(e) => {
                      if (!isActive(link.to))
                        (e.currentTarget as HTMLElement).style.color = "#ffffff";
                    }}
                    onMouseLeave={(e) => {
                      if (!isActive(link.to))
                        (e.currentTarget as HTMLElement).style.color =
                          "rgba(255,255,255,0.75)";
                    }}
                  >
                    {link.label}
                  </Link>
                ))}
                {/* Contact us with arrow icon */}
                <a
                  href="mailto:contact@echovoice.ai"
                  className="flex items-center gap-1 px-3 py-1.5 rounded-full text-sm transition-all duration-200 whitespace-nowrap"
                  style={{ color: "rgba(255,255,255,0.75)" }}
                  onMouseEnter={(e) =>
                    ((e.currentTarget as HTMLElement).style.color = "#ffffff")
                  }
                  onMouseLeave={(e) =>
                    ((e.currentTarget as HTMLElement).style.color =
                      "rgba(255,255,255,0.75)")
                  }
                >
                  Contact us
                  <ArrowUpRight size={13} strokeWidth={2.5} />
                </a>
              </div>
            </div>

            {/* ── Auth Controls ─────────────────────────────────────── */}
            <div className="flex items-center gap-2">
              {currentUser ? (
                <>
                  <button
                    onClick={onOpenAuth}
                    className="hidden sm:flex items-center gap-2 px-3 py-1.5 rounded-full text-sm text-white/80 hover:text-white transition-colors duration-200"
                    aria-label="Account settings"
                    style={{
                      background: "rgba(255,255,255,0.07)",
                      border: "1px solid rgba(255,255,255,0.12)",
                      minHeight: 44,
                    }}
                  >
                    <User size={14} />
                    <span className="max-w-[120px] truncate">
                      {currentUser.full_name || currentUser.email}
                    </span>
                  </button>
                  <button
                    onClick={onSignOut}
                    className="flex items-center justify-center w-10 h-10 rounded-full text-white/60 hover:text-white transition-colors duration-200"
                    style={{
                      background: "rgba(255,255,255,0.06)",
                      border: "1px solid rgba(255,255,255,0.1)",
                      minHeight: 44,
                      minWidth: 44,
                    }}
                    aria-label="Sign out"
                  >
                    <LogOut size={15} />
                  </button>
                </>
              ) : (
                <button
                  onClick={onOpenAuth}
                  className="hidden sm:flex items-center gap-2 px-4 py-2 rounded-full text-sm text-white font-medium transition-all duration-200 hover:bg-white/10"
                  style={{
                    background: "rgba(255,255,255,0.08)",
                    border: "1px solid rgba(255,255,255,0.15)",
                    minHeight: 44,
                  }}
                  id="nav-sign-in-btn"
                  aria-label="Sign in"
                >
                  <LogIn size={14} />
                  Sign In
                </button>
              )}

              {/* ── Hamburger (mobile) ─────────────────────────────── */}
              <button
                onClick={() => setDrawerOpen(true)}
                className="lg:hidden flex items-center justify-center w-11 h-11 rounded-full text-white/80 hover:text-white transition-colors"
                style={{
                  background: "rgba(255,255,255,0.06)",
                  border: "1px solid rgba(255,255,255,0.1)",
                }}
                aria-label="Open navigation menu"
                aria-expanded={drawerOpen}
              >
                <Menu size={20} />
              </button>
            </div>
          </div>
        </div>
      </nav>

      {/* ── Mobile Drawer ─────────────────────────────────────────── */}
      <AnimatePresence>
        {drawerOpen && (
          <>
            {/* Backdrop */}
            <motion.div
              className="fixed inset-0 z-40 bg-black/70"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              onClick={() => setDrawerOpen(false)}
              aria-hidden="true"
            />
            {/* Drawer panel */}
            <motion.div
              className="fixed top-0 right-0 bottom-0 z-50 w-72 flex flex-col"
              style={{
                background: "rgba(5, 8, 18, 0.97)",
                backdropFilter: "blur(24px)",
                borderLeft: "1px solid rgba(255,255,255,0.08)",
              }}
              initial={{ x: "100%" }}
              animate={{ x: 0 }}
              exit={{ x: "100%" }}
              transition={{ type: "spring", stiffness: 320, damping: 32 }}
              role="dialog"
              aria-label="Navigation drawer"
            >
              {/* Drawer header */}
              <div className="flex items-center justify-between px-6 py-5">
                <span className="text-white font-semibold text-base">
                  EchoVoice
                </span>
                <button
                  onClick={() => setDrawerOpen(false)}
                  className="flex items-center justify-center w-10 h-10 rounded-full text-white/60 hover:text-white transition-colors"
                  style={{ background: "rgba(255,255,255,0.07)" }}
                  aria-label="Close menu"
                >
                  <X size={18} />
                </button>
              </div>

              {/* Drawer links */}
              <nav className="flex-1 px-4 pb-6 space-y-1 overflow-y-auto">
                {NAV_LINKS.map((link) => (
                  <Link
                    key={link.to}
                    to={link.to}
                    onClick={() => setDrawerOpen(false)}
                    className="flex items-center px-4 py-3.5 rounded-xl text-base font-medium transition-all duration-200"
                    style={{
                      color: isActive(link.to) ? "#ffffff" : "rgba(255,255,255,0.7)",
                      background: isActive(link.to)
                        ? "rgba(255,255,255,0.08)"
                        : "transparent",
                      minHeight: 44,
                    }}
                  >
                    {link.label}
                  </Link>
                ))}
                <a
                  href="mailto:contact@echovoice.ai"
                  className="flex items-center gap-2 px-4 py-3.5 rounded-xl text-base font-medium text-white/70 hover:text-white transition-colors"
                  style={{ minHeight: 44 }}
                >
                  Contact us
                  <ArrowUpRight size={15} />
                </a>

                {/* Auth in drawer */}
                <div className="pt-4 border-t border-white/10 mt-4">
                  {currentUser ? (
                    <div className="space-y-2">
                      <div className="px-4 py-2 text-sm text-white/50">
                        {currentUser.email}
                      </div>
                      <button
                        onClick={() => {
                          onSignOut();
                          setDrawerOpen(false);
                        }}
                        className="flex items-center gap-2 w-full px-4 py-3 rounded-xl text-sm text-white/70 hover:text-white transition-colors"
                        style={{
                          background: "rgba(255,255,255,0.05)",
                          minHeight: 44,
                        }}
                      >
                        <LogOut size={14} /> Sign Out
                      </button>
                    </div>
                  ) : (
                    <button
                      onClick={() => {
                        onOpenAuth();
                        setDrawerOpen(false);
                      }}
                      className="flex items-center gap-2 w-full px-4 py-3 rounded-xl text-sm text-white font-medium transition-all"
                      style={{
                        background: "rgba(100, 206, 251, 0.12)",
                        border: "1px solid rgba(100, 206, 251, 0.25)",
                        minHeight: 44,
                      }}
                    >
                      <LogIn size={14} /> Sign In / Demo
                    </button>
                  )}
                </div>
              </nav>
            </motion.div>
          </>
        )}
      </AnimatePresence>
    </>
  );
}
