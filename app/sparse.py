"""BM25-style sparse vectors for Vietnamese, without extra dependencies.

Vietnamese words are separated by spaces but compounds are 2+ syllables ("mức lương", "tối thiểu"),
so we index syllable unigrams AND bigrams. Tokens are hashed to integer ids; Qdrant applies the IDF
part server-side (Modifier.IDF), we supply the saturated term-frequency part.
"""
import re
import unicodedata
import zlib
from collections import Counter

TOKEN_RE = re.compile(r"\w+", re.UNICODE)
DIM = 2**20
K1, B = 1.2, 0.75
AVG_DOC_LEN = 400.0  # rough average of (unigrams + bigrams) per chunk; only used for length normalisation


def tokenize(text: str) -> list[str]:
    words = TOKEN_RE.findall(unicodedata.normalize("NFC", text).lower())
    return words + [f"{a} {b}" for a, b in zip(words, words[1:])]


def _index(token: str) -> int:
    return zlib.crc32(token.encode("utf-8")) % DIM


def doc_vector(text: str) -> tuple[list[int], list[float]]:
    """Document side: saturated term frequency with length normalisation."""
    counts: Counter[int] = Counter(_index(t) for t in tokenize(text))
    length = sum(counts.values()) or 1
    norm = K1 * (1 - B + B * length / AVG_DOC_LEN)
    indices = sorted(counts)
    return indices, [counts[i] * (K1 + 1) / (counts[i] + norm) for i in indices]


def query_vector(text: str) -> tuple[list[int], list[float]]:
    """Query side: every distinct term counts once."""
    indices = sorted({_index(t) for t in tokenize(text)})
    return indices, [1.0] * len(indices)
