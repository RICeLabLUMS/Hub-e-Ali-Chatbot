"""
Extract numeric citations from indexed content so the chat layer can produce
references like "Quran 8:5", "Sermon 17", "Al-Kafi Vol. 8 H. 245" instead of
opaque chunk_ids.

Two layers of metadata:

  doc-level (set once per source from its title/filename, inherited by all
             chunks of that doc):
    - chapter_num    e.g. 8 from "AL-ANFAAL (Chapter 8) Verses 1-40"
    - verse_range    e.g. "1-40" from the same title
    - volume         e.g. 8 from "AlKafi-Volume8-Urdu.pdf"

  chunk-level (re-derived per chunk after chunker.chunk_pages splits the text):
    - refs_quran     e.g. ["8:1", "8:5-8", "2:21"]. Validated against per-surah
                     ayah counts so "5:150" is rejected (Al-Ma'idah has 120 ayat),
                     and Arabic/Persian digits (٠-٩, ۰-۹) are normalized to ASCII.
    - section_title  e.g. "VERSE 1" / "Sermon 17" / "آیت 1" / "حدیث 245".
                     The first section header visible near the top of the chunk;
                     Latin AND Arabic/Urdu patterns recognized.
    - hadith_refs    e.g. ["Vol. 1 H. 245", "Hadith 245", "Sermon 17"]
                     Multiple shapes; safe-by-default - won't tag random numbers.

Design choices:
  - Quran refs use per-surah ayah caps (table below) so "5:150" doesn't slip
    through as a false positive even though it's within global limits.
  - section_title is the FIRST heading the chunk contains (the section the
    chunk *belongs to*); we don't try to enumerate all section labels.
  - Hadith ref patterns require an explicit keyword (Hadith / Sermon / Letter /
    Saying) or a Vol+H combo - never bare numbers.
  - No external taxonomies / databases / LLMs - regex only - so this stays
    cheap and deterministic.
"""

from __future__ import annotations

import re
from typing import Optional

__all__ = [
    "extract_quran_refs",
    "extract_chapter_from_title",
    "extract_verse_range_from_title",
    "extract_volume",
    "extract_section_title",
    "extract_hadith_refs",
]

# ----- Quran reference data -----

# Number of ayat per surah, 1-indexed. Slot 0 is a placeholder so that
# _SURAH_AYAH_COUNTS[surah_num] works directly. Sourced from the standard
# Hafs reading - the count canonical across the Islamic mainstream.
_SURAH_AYAH_COUNTS: tuple[int, ...] = (
    0,
    7, 286, 200, 176, 120, 165, 206, 75, 129, 109,
    123, 111, 43, 52, 99, 128, 111, 110, 98, 135,
    112, 78, 118, 64, 77, 227, 93, 88, 69, 60,
    34, 30, 73, 54, 45, 83, 182, 88, 75, 85,
    54, 53, 89, 59, 37, 35, 38, 29, 18, 45,
    60, 49, 62, 55, 78, 96, 29, 22, 24, 13,
    14, 11, 11, 18, 12, 12, 30, 52, 52, 44,
    28, 28, 20, 56, 40, 31, 50, 40, 46, 42,
    29, 19, 36, 25, 22, 17, 19, 26, 30, 20,
    15, 21, 11, 8, 8, 19, 5, 8, 8, 11,
    11, 8, 3, 9, 5, 4, 7, 3, 6, 3,
    5, 4, 5, 6,
)
assert len(_SURAH_AYAH_COUNTS) == 115, "expected 0 + 114 surahs"

_MAX_SURAH = 114  # convenience cap


# ----- Digit normalization (Arabic-Indic / Persian -> ASCII) -----

_DIGIT_NORMALIZE_MAP = str.maketrans({
    # Arabic-Indic
    "٠": "0", "١": "1", "٢": "2", "٣": "3", "٤": "4",
    "٥": "5", "٦": "6", "٧": "7", "٨": "8", "٩": "9",
    # Eastern Arabic-Indic (Persian/Urdu)
    "۰": "0", "۱": "1", "۲": "2", "۳": "3", "۴": "4",
    "۵": "5", "۶": "6", "۷": "7", "۸": "8", "۹": "9",
})


def _normalize_digits(text: str) -> str:
    """Replace Arabic-Indic and Persian digits with ASCII 0-9. All regex below
    operate on the result so the same pattern catches '8:5' and '۸:۵'."""
    return text.translate(_DIGIT_NORMALIZE_MAP)


# ----- Quran refs in body text -----

# Single bare ref or ranged: 8:1 / 8:5-8 / 8:5–8 (en-dash).
# Bracketing is allowed but not required. Refs preceded or followed by another
# digit/colon are skipped to avoid matching mid-number ("8.5:1" -> no match).
_RE_QURAN_REF = re.compile(
    r"""
    (?<![\d:.\-/])
    (\d{1,3})
    :
    (\d{1,3})
    (?:\s*[-–]\s*(\d{1,3}))?
    (?![\d:])
    """,
    re.VERBOSE,
)


def extract_quran_refs(text: str) -> list[str]:
    """
    Find Quran chapter:verse references in chunk text.

    Returns canonical strings like ["8:1", "8:5-8", "2:21"]. Validates against
    the per-surah ayah count table so out-of-range pairs (5:150, 1024:7) are
    dropped silently. Arabic-Indic / Persian digits are normalized before
    matching, so "۸:۵" and "8:5" produce the same ref.
    """
    if not text:
        return []
    text = _normalize_digits(text)
    seen: set[str] = set()
    out: list[str] = []
    for m in _RE_QURAN_REF.finditer(text):
        surah = int(m.group(1))
        ayah_start = int(m.group(2))
        ayah_end_raw = m.group(3)
        ayah_end = int(ayah_end_raw) if ayah_end_raw else None

        if not (1 <= surah <= _MAX_SURAH):
            continue
        max_ayah = _SURAH_AYAH_COUNTS[surah]
        if not (1 <= ayah_start <= max_ayah):
            continue
        if ayah_end is not None and not (ayah_start < ayah_end <= max_ayah):
            continue

        canonical = f"{surah}:{ayah_start}" + (f"-{ayah_end}" if ayah_end else "")
        if canonical not in seen:
            seen.add(canonical)
            out.append(canonical)
    return out


# ----- Title parsing -----

_RE_CHAPTER_IN_TITLE = re.compile(r"\bchapter\s+(\d{1,3})\b", re.IGNORECASE)
_RE_VERSES_IN_TITLE = re.compile(
    r"\bverses?\s+(\d{1,3})\s*[-–]\s*(\d{1,3})\b",
    re.IGNORECASE,
)
_RE_SINGLE_VERSE_IN_TITLE = re.compile(r"\bverse\s+(\d{1,3})\b", re.IGNORECASE)


def extract_chapter_from_title(title: Optional[str]) -> Optional[int]:
    """Pull a leading "Chapter N" from a doc title. Range-checked against
    the 114-surah limit so post titles with random chapter numbers still
    pass through (but Quran surah numbers are valid)."""
    if not title:
        return None
    m = _RE_CHAPTER_IN_TITLE.search(_normalize_digits(title))
    if not m:
        return None
    n = int(m.group(1))
    return n if 1 <= n <= _MAX_SURAH else None


def extract_verse_range_from_title(title: Optional[str]) -> Optional[str]:
    """Pull "Verses M-N" or "Verse N" from a doc title. Canonical string
    output ("5-8" or "1") or None."""
    if not title:
        return None
    t = _normalize_digits(title)
    m = _RE_VERSES_IN_TITLE.search(t)
    if m:
        start, end = int(m.group(1)), int(m.group(2))
        if 1 <= start < end <= max(_SURAH_AYAH_COUNTS):
            return f"{start}-{end}"
    m2 = _RE_SINGLE_VERSE_IN_TITLE.search(t)
    if m2:
        v = int(m2.group(1))
        if 1 <= v <= max(_SURAH_AYAH_COUNTS):
            return str(v)
    return None


# ----- Volume from filename / title -----

_RE_VOLUME = re.compile(
    r"\b(?:vol(?:ume)?\.?\s*[-_]?\s*|v)(\d{1,3})\b",
    re.IGNORECASE,
)


def extract_volume(filename_or_title: Optional[str]) -> Optional[int]:
    """Pull a volume number from a PDF filename or media title."""
    if not filename_or_title:
        return None
    m = _RE_VOLUME.search(_normalize_digits(filename_or_title))
    if not m:
        return None
    n = int(m.group(1))
    return n if 1 <= n <= 999 else None


# ----- Section title from chunk body (Latin + RTL scripts) -----

_RE_SECTION = re.compile(
    r"""
    ^\s*
    (?:\#{1,6}\s*)?                              # optional markdown heading prefix
    (
        # English / Latin script
        VERSES?\s+\d{1,3}(?:\s*[-–]\s*\d{1,3})?
      | Sermon\s+\d{1,3}
      | Letter\s+\d{1,3}
      | Saying\s+\d{1,3}
      | Hadith\s+\d{1,5}
      | Chapter\s+\d{1,3}
      | Section\s+\d{1,3}

        # Urdu / Arabic - matched against digit-normalized text, so the
        # source can use Arabic-Indic numerals freely.
      | آیت\s+\d{1,3}(?:\s*[-–]\s*\d{1,3})?   # آیت (Urdu Ayat)
      | آية\s+\d{1,3}(?:\s*[-–]\s*\d{1,3})?   # آية (Arabic Ayah)
      | سورة\s+\d{1,3}                        # سورة (Arabic Surah)
      | سورۃ\s+\d{1,3}                        # سورۃ (Urdu Surah)
      | حدیث\s+\d{1,5}                        # حدیث (Urdu Hadith)
      | حديث\s+\d{1,5}                        # حديث (Arabic Hadith)
      | خطبہ\s+\d{1,3}                        # خطبہ (Urdu Khutba)
      | خطبة\s+\d{1,3}                        # خطبة (Arabic Khutba)
      | باب\s+\d{1,3}                              # باب (Bab / Chapter)
      | فصل\s+\d{1,3}                              # فصل (Fasl / Section)
    )
    """,
    re.MULTILINE | re.VERBOSE | re.IGNORECASE,
)


def extract_section_title(chunk_text: str) -> Optional[str]:
    """Return the FIRST section header found in this chunk - Latin or RTL."""
    if not chunk_text:
        return None
    normalized = _normalize_digits(chunk_text)
    m = _RE_SECTION.search(normalized)
    if not m:
        return None
    header = re.sub(r"\s+", " ", m.group(1)).strip()
    header = header.replace("–", "-")  # en-dash -> hyphen
    return header


# ----- Hadith / Sermon / Letter / Saying refs -----

# Vol N H N (with various separators) - most common explicit hadith ref shape.
_RE_HADITH_VOL_H = re.compile(
    r"\bVol(?:ume)?\.?\s*(\d{1,3})\s*[,;:\s]+\s*H(?:adith)?\.?\s*(\d{1,5})\b",
    re.IGNORECASE,
)
# Standalone "Hadith N" - requires the full word to avoid noise.
_RE_HADITH_WORD = re.compile(r"\bHadith\s+(\d{1,5})\b", re.IGNORECASE)
# Sermon N / Letter N / Saying N - Nahjul Balagha style.
_RE_NAHJ_REF = re.compile(
    r"\b(Sermon|Letter|Saying)\s+(\d{1,3})\b",
    re.IGNORECASE,
)


def extract_hadith_refs(text: str) -> list[str]:
    """
    Extract hadith / sermon / letter / saying reference numbers.

    Returns canonical short strings:
      ["Vol. 1 H. 245", "Hadith 245", "Sermon 17", "Letter 31", "Saying 7"]

    Conservative on purpose: requires an explicit keyword. Never tags bare
    numbers as hadith refs. Digit normalization runs first so Arabic-Indic
    numerals in the source text get picked up too.
    """
    if not text:
        return []
    normalized = _normalize_digits(text)
    refs: set[str] = set()

    for m in _RE_HADITH_VOL_H.finditer(normalized):
        refs.add(f"Vol. {m.group(1)} H. {m.group(2)}")
    for m in _RE_HADITH_WORD.finditer(normalized):
        refs.add(f"Hadith {m.group(1)}")
    for m in _RE_NAHJ_REF.finditer(normalized):
        kind = m.group(1).title()
        refs.add(f"{kind} {m.group(2)}")

    return sorted(refs)
