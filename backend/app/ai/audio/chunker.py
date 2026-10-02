"""
Text Chunker for Long-Form Speech (backend/app/ai/audio/chunker.py)
-------------------------------------------------------------------
PYTHON CONCEPTS DEMONSTRATED:
- Dataclasses: Represents individual sentence/phrase chunks with timing metadata.
- Regular Expressions (`re`): Handles complex multilingual sentence boundaries,
  including Western punctuation (. ! ?) and Indian language punctuation (। Purna Viram, ।। Deergh Viram).
- Text Partitioning & Windowing: Enforces a ~250 character ceiling to prevent
  neural hallucination, repetition loops, and tempo drift in XTTS-v2 / zero-shot cloners.
"""

import re
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass(frozen=True)
class ChunkItem:
    """
    Represents an atomic speech synthesis unit with boundary pause metadata.
    """
    index: int
    text: str
    pause_after_ms: int        # Natural pause length after chunk (e.g. 250ms, 500ms, 800ms)
    char_count: int
    word_count: int

    @property
    def estimated_duration_sec(self) -> float:
        """Rough duration estimation (150 words per minute ~ 2.5 words/sec)."""
        return max(0.5, (self.word_count / 2.5) + (self.pause_after_ms / 1000.0))


class ScriptChunker:
    """
    Splits long scripts, articles, or transcripts into natural phonetic chunks.
    Ensures seamless prosody and prevents neural model degradation on long sequences.
    """

    # Matches sentence endings across Western and Indian scripts:
    # Western: . ! ? ; \n
    # Indian / Devanagari / Bengali / Odia: । (U+0964 Purna Viram), ।। (U+0965 Deergh Viram)
    SENTENCE_ENDINGS = r'(?<=[.!?।॥\n])\s+'
    CLAUSE_SPLITTERS = r'(?<=[,;:\—\–\-])\s+'

    def __init__(self, max_chars: int = 250, min_chars: int = 30):
        self.max_chars = max_chars
        self.min_chars = min_chars

    def chunk_text(self, text: str) -> List[ChunkItem]:
        """
        Splits text into an ordered list of ChunkItems with natural pause timings.
        """
        if not text or not text.strip():
            return []

        # 1. Clean markdown headers, bullet marks, and excessive whitespace
        normalized = self._normalize_text(text)

        # 2. Split into preliminary paragraphs
        paragraphs = [p.strip() for p in normalized.split("\n\n") if p.strip()]

        raw_chunks: List[tuple[str, int]] = []

        for p_idx, paragraph in enumerate(paragraphs):
            # Split paragraph into sentences
            sentences = [s.strip() for s in re.split(self.SENTENCE_ENDINGS, paragraph) if s.strip()]

            for s_idx, sentence in enumerate(sentences):
                is_last_in_paragraph = (s_idx == len(sentences) - 1)

                # Determine natural pause length based on boundary type
                if is_last_in_paragraph:
                    pause_ms = 800  # Paragraph transition
                elif sentence.endswith(("?", "!")):
                    pause_ms = 600  # Question / Exclamation emphasis
                elif sentence.endswith(("॥", "।।")):
                    pause_ms = 750  # Deergh Viram
                elif sentence.endswith(("।")):
                    pause_ms = 500  # Purna Viram
                else:
                    pause_ms = 500  # Standard period

                # If sentence exceeds max_chars, split on clauses or word boundaries
                if len(sentence) > self.max_chars:
                    sub_chunks = self._subdivide_long_sentence(sentence)
                    for sub_idx, sub in enumerate(sub_chunks):
                        sub_pause = pause_ms if (sub_idx == len(sub_chunks) - 1) else 250
                        raw_chunks.append((sub, sub_pause))
                else:
                    raw_chunks.append((sentence, pause_ms))

        # 3. Merge micro-chunks (< min_chars) with adjacent chunks when possible
        merged_chunks = self._merge_short_chunks(raw_chunks)

        # 4. Construct final typed ChunkItems
        result: List[ChunkItem] = []
        for idx, (chunk_str, pause) in enumerate(merged_chunks):
            words = chunk_str.split()
            result.append(
                ChunkItem(
                    index=idx,
                    text=chunk_str,
                    pause_after_ms=pause,
                    char_count=len(chunk_str),
                    word_count=len(words)
                )
            )

        return result

    def _normalize_text(self, text: str) -> str:
        """Strips markdown formatting, bold/italics, and unifies punctuation."""
        t = re.sub(r'#+\s*', '', text)             # Remove markdown headers
        t = re.sub(r'\*{1,3}(.*?)\*{1,3}', r'\1', t) # Strip bold / italic stars
        t = re.sub(r'`(.*?)`', r'\1', t)           # Strip inline code
        t = re.sub(r'\[(.*?)\]\(.*?\)', r'\1', t)  # Replace markdown links with link text
        t = re.sub(r'\r\n', '\n', t)               # Normalize Windows CRLF
        t = re.sub(r'[ \t]+', ' ', t)              # Collapse horizontal whitespace
        return t.strip()

    def _subdivide_long_sentence(self, sentence: str) -> List[str]:
        """Subdivides a long sentence along clause boundaries (commas, semicolons)."""
        clauses = [c.strip() for c in re.split(self.CLAUSE_SPLITTERS, sentence) if c.strip()]
        if not clauses:
            clauses = [sentence]

        chunks: List[str] = []
        current: List[str] = []
        current_len = 0

        for clause in clauses:
            if current_len + len(clause) + 1 <= self.max_chars:
                current.append(clause)
                current_len += len(clause) + 1
            else:
                if current:
                    chunks.append(", ".join(current))
                current = [clause]
                current_len = len(clause)

        if current:
            chunks.append(", ".join(current))

        # Fallback if any single clause is still too long: split on space
        final_subs: List[str] = []
        for chk in chunks:
            if len(chk) <= self.max_chars:
                final_subs.append(chk)
            else:
                # Word-level wrap
                words = chk.split()
                line: List[str] = []
                line_len = 0
                for w in words:
                    if line_len + len(w) + 1 <= self.max_chars:
                        line.append(w)
                        line_len += len(w) + 1
                    else:
                        if line:
                            final_subs.append(" ".join(line))
                        line = [w]
                        line_len = len(w)
                if line:
                    final_subs.append(" ".join(line))

        return final_subs

    def _merge_short_chunks(self, raw_chunks: List[tuple[str, int]]) -> List[tuple[str, int]]:
        """Merges unnaturally brief sentence fragments."""
        if not raw_chunks:
            return []

        merged: List[tuple[str, int]] = []
        for text, pause in raw_chunks:
            if merged and len(merged[-1][0]) + len(text) + 1 <= self.max_chars and len(merged[-1][0]) < self.min_chars:
                prev_text, _ = merged.pop()
                merged.append((f"{prev_text} {text}", pause))
            else:
                merged.append((text, pause))

        return merged


chunker = ScriptChunker()
