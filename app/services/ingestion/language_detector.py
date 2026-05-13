import re
from lingua import Language, LanguageDetectorBuilder

# Detector limited to our three target languages — faster and more accurate
# than scanning lingua's full ~75-language model.
_detector = (
    LanguageDetectorBuilder
    .from_languages(Language.ENGLISH, Language.ARABIC, Language.URDU)
    .with_minimum_relative_distance(0.1)
    .build()
)

LANG_CODE_MAP = {
    Language.ENGLISH: "en",
    Language.ARABIC: "ar",
    Language.URDU: "ur",
}

LINGUA_MIN_LEN = 25   # lingua is unreliable below ~25 chars; below that use script fallback

URDU_MARKERS = ["ہے", "کا", "کی", "میں", "سے", "کو", "نے", "ہیں", "تھا", "تھی"]
ARABIC_MARKERS = ["الذي", "التي", "على", "في", "من", "إلى", "عن", "هذا", "هذه", "كان"]
URDU_ONLY_CHARS = "ٹڈڑںےہکگچپژ"


def detect_language(text: str) -> str:
    """Return one of {'en', 'ar', 'ur', 'unknown'}."""
    if not text:
        return "unknown"
    stripped = text.strip()
    if len(stripped) < 10:
        return "unknown"

    # For very short text, lingua hallucinates — go straight to script detection.
    if len(stripped) < LINGUA_MIN_LEN:
        return _detect_by_script(stripped)

    result = _detector.detect_language_of(stripped)
    if result is not None:
        code = LANG_CODE_MAP.get(result, "unknown")
        if code != "unknown":
            return code

    return _detect_by_script(stripped)


def _detect_by_script(text: str) -> str:
    arabic_chars = len(re.findall(r"[؀-ۿ]", text))
    latin_chars = len(re.findall(r"[A-Za-z]", text))
    total = len(text.replace(" ", "")) or 1

    if arabic_chars / total > 0.4:
        # Arabic and Urdu share script — distinguish by markers + Urdu-only chars
        urdu_only_hits = sum(1 for c in text if c in URDU_ONLY_CHARS)
        urdu_score = sum(1 for m in URDU_MARKERS if m in text) + urdu_only_hits
        arabic_score = sum(1 for m in ARABIC_MARKERS if m in text)
        return "ur" if urdu_score >= arabic_score and urdu_score > 0 else "ar"

    if latin_chars / total > 0.4:
        return "en"

    return "unknown"


def split_by_language(text: str) -> list[tuple[str, str]]:
    """
    Split text into runs of contiguous same-language paragraphs.

    Returns: [(segment_text, lang), ...]

    Used by the chunker to handle mixed-language pages (Arabic paragraph
    followed by English translation, etc.) so each segment gets its own
    language tag and is chunked in isolation.
    """
    if not text or not text.strip():
        return []

    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    if not paragraphs:
        return [(text.strip(), detect_language(text))]

    tagged = [(p, detect_language(p)) for p in paragraphs]

    # Merge consecutive paragraphs that share a language
    merged: list[tuple[str, str]] = []
    for segment, lang in tagged:
        if merged and merged[-1][1] == lang:
            merged[-1] = (merged[-1][0] + "\n\n" + segment, lang)
        else:
            merged.append((segment, lang))

    return merged


def detect_dominant_language(texts: list[str]) -> str:
    from collections import Counter
    langs = [detect_language(t) for t in texts]
    most_common = Counter(langs).most_common(1)
    return most_common[0][0] if most_common else "unknown"
