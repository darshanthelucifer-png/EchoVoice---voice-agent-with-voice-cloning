/**
 * ShinyText — Framer Motion animated gradient text sweep (frontend/src/components/ui/ShinyText.tsx)
 * --------------------------------------------------------------------------------------------------
 * Animates a diagonal gradient highlight (shine) sweeping left-to-right over the text continuously.
 * All colours, speed and spread are driven by CSS variables so the AudioThemeProvider can
 * override them reactively without re-rendering the component.
 *
 * Props:
 *   text       — The string to display.
 *   baseColor  — Default fill colour (overridden by --accent CSS var when theme is active).
 *   shineColor — Highlight peak colour (overridden by --shine CSS var when theme is active).
 *   speed      — Full sweep duration in seconds (overridden by --bpm-duration when theme is active).
 *   spread     — Gradient angle in degrees.
 */

import { motion } from "framer-motion";

interface ShinyTextProps {
  text: string;
  baseColor?: string;
  shineColor?: string;
  speed?: number;      // seconds for one full sweep
  spread?: number;     // gradient angle in degrees
  className?: string;
}

export function ShinyText({
  text,
  baseColor = "#64CEFB",
  shineColor = "#ffffff",
  speed = 3,
  spread = 100,
  className = "",
}: ShinyTextProps) {
  const gradientImage = `linear-gradient(${spread}deg, ${baseColor} 0%, ${baseColor} 35%, ${shineColor} 50%, ${baseColor} 65%, ${baseColor} 100%)`;

  return (
    <motion.span
      className={className}
      style={{
        backgroundImage: gradientImage,
        backgroundSize: "200% auto",
        backgroundClip: "text",
        WebkitBackgroundClip: "text",
        WebkitTextFillColor: "transparent",
        display: "inline-block",
      }}
      animate={{ backgroundPosition: ["150% center", "-50% center"] }}
      transition={{
        duration: speed,
        ease: "linear",
        repeat: Infinity,
        repeatType: "loop",
      }}
    >
      {text}
    </motion.span>
  );
}
