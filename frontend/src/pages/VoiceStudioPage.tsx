/**
 * VoiceStudioPage — Voice Studio page wrapper (frontend/src/pages/VoiceStudioPage.tsx)
 * --------------------------------------------------------------------------------------
 * Wraps the existing VoiceStudio component (Phases 1-4) inside the new design shell.
 * All existing functionality (Record/Upload → Clean → Profile → Script → Generate → Download,
 * A/B players, similarity scores, Tier picker, RVC training trigger) is preserved.
 */

import { VoiceStudio } from "../components/VoiceStudio";
import type { UserProfile } from "../services/api";

interface VoiceStudioPageProps {
  currentUser: UserProfile | null;
  onOpenAuth: () => void;
}

export function VoiceStudioPage({ currentUser, onOpenAuth }: VoiceStudioPageProps) {
  return (
    <div
      className="min-h-screen pt-16"
      style={{ background: "linear-gradient(180deg, #000 0%, #050818 100%)" }}
    >
      <VoiceStudio currentUser={currentUser} onOpenAuth={onOpenAuth} />
    </div>
  );
}
