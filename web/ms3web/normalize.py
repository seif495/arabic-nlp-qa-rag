from __future__ import annotations

import re

_ARABIC_DIACRITICS = re.compile(r"[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06ED]")
_ALIF_VARIANTS = str.maketrans({"أ": "ا", "إ": "ا", "آ": "ا", "ى": "ي"})


def arabic_post_normalize(text: str) -> str:
    """Apply the ADR §1.8 Arabic post-normalization used by all MS2 metrics."""
    normalized = text.translate(_ALIF_VARIANTS)
    normalized = _ARABIC_DIACRITICS.sub("", normalized)
    return " ".join(normalized.split())
