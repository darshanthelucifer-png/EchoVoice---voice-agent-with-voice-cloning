/**
 * EchoVoice App Shell with React Router (frontend/src/App.tsx)
 * ------------------------------------------------------------
 * Phase 7: Full routing shell replacing the old tab-based layout.
 * Routes: / Home, /talk Talk, /voice-studio Voice Studio,
 *         /song-studio Song Studio, /knowledge Knowledge,
 *         /dashboard Dashboard, /settings Settings.
 * Auth state is managed here and threaded down via props.
 * AuthModal is lifted to App level so it can be opened from any page.
 */

import { useState, useEffect } from "react";
import { BrowserRouter, Routes, Route } from "react-router-dom";
import { Navbar } from "./components/layout/Navbar";
import { AuthModal } from "./components/AuthModal";
import { HomePage } from "./pages/HomePage";
import { TalkPage } from "./pages/TalkPage";
import { VoiceStudioPage } from "./pages/VoiceStudioPage";
import { SongStudioPage } from "./pages/SongStudioPage";
import { KnowledgePage } from "./pages/KnowledgePage";
import { DashboardPage } from "./pages/DashboardPage";
import { SettingsPage } from "./pages/SettingsPage";
import {
  fetchCurrentUser,
  ensureAuthenticatedSession,
  clearAuthToken,
} from "./services/api";
import type { UserProfile } from "./services/api";

export function App() {
  const [currentUser, setCurrentUser] = useState<UserProfile | null>(null);
  const [isAuthModalOpen, setIsAuthModalOpen] = useState(false);

  // Auto-bootstrap session on mount
  useEffect(() => {
    const init = async () => {
      try {
        await ensureAuthenticatedSession();
        const user = await fetchCurrentUser();
        if (user) setCurrentUser(user);
      } catch (e) {
        console.warn("Session init:", e);
      }
    };
    init();
  }, []);

  const handleSignOut = () => {
    clearAuthToken();
    setCurrentUser(null);
  };

  return (
    <BrowserRouter>
      {/* Global Navbar (fixed, above all pages) */}
      <Navbar
        currentUser={currentUser}
        onOpenAuth={() => setIsAuthModalOpen(true)}
        onSignOut={handleSignOut}
      />

      {/* Page Routes */}
      <Routes>
        <Route path="/" element={<HomePage />} />
        <Route path="/talk" element={<TalkPage />} />
        <Route
          path="/voice-studio"
          element={
            <VoiceStudioPage
              currentUser={currentUser}
              onOpenAuth={() => setIsAuthModalOpen(true)}
            />
          }
        />
        <Route path="/song-studio" element={<SongStudioPage />} />
        <Route path="/knowledge" element={<KnowledgePage />} />
        <Route path="/dashboard" element={<DashboardPage />} />
        <Route path="/settings" element={<SettingsPage />} />
        {/* Fallback: redirect unknown paths to home */}
        <Route path="*" element={<HomePage />} />
      </Routes>

      {/* Global Auth Modal (accessible from any page) */}
      <AuthModal
        isOpen={isAuthModalOpen}
        onClose={() => setIsAuthModalOpen(false)}
        onAuthSuccess={(user) => setCurrentUser(user)}
      />
    </BrowserRouter>
  );
}

export default App;
