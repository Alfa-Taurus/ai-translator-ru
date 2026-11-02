#!/usr/bin/env python3
"""
AI Literary Novel Translator — Cerberus Enterprise v10.0 (Clean Architecture)

Pipeline Stages & Cost Routing:
- Stage 0.5 (Annotation Cleaner): deepseek-v4.1-flash (x0.3) — Anchor-based A/N & meta-text slicing (zero regex).
- Stage 1.0 (Dynamic NER Harvester): deepseek-v4.1-flash (x0.3) — ~6k token super-batches with auto-bisection.
- Stage 2.0 (Lore Curator): claude-sonnet-5 (x2.0) — Full Russian morphology (lemma, surface, gender, animacy).
- Stage 2.5 (Two-Tier Typo Normalizer):
    * Tier 1: Local 370k English Lexicon (O(1) lookup, 0 tokens).
    * Tier 2: gpt-6-luna (x0.38) binary typo verification.
- Stage 3.0 (Core Literary Translator): claude-sonnet-5 (x2.0) — SSE streaming prose with Longest-Match glossary masking.
- Diagnostic Stop: Immediate halt with GUI trigger dump on censorship/refusal.
"""

from __future__ import annotations

import argparse
import asyncio
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from enum import Enum
import hashlib
import html
import json
import logging
import os
from pathlib import Path
import random
import re
import tempfile
import time
from typing import Any, AsyncIterator, Callable, Dict, List, Optional, Set, Tuple
import urllib.request

from bs4 import BeautifulSoup
from bs4.element import NavigableString, Tag
import ebooklib
from ebooklib import epub
import httpx
import json_repair

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("TranslatorPipeline")

PROMPT_VERSION = "10.0.0"
LORE_SCHEMA_VERSION = 10
MAX_LORE_BATCH_ATTEMPTS = 4
TYPO_MAX_HITS = 2
HARVEST_SATURATION_LIMIT = 45
RU_VOWELS = set("аеёиоуыэюя")
WORDS_ALPHA_URL = "https://raw.githubusercontent.com/dwyl/english-words/master/words_alpha.txt"

CERBERUS_MODELS = [
    # Premium Literary & Curation (x2.0 - x4.0)
    "claude-sonnet-5",
    "claude-sonnet-4-6",
    "gpt-6.1-sol",
    "gpt-6-sol",
    "claude-opus-5",
    "claude-opus-5-5",
    "gemini-3.8-flash",
    # Workhorses & Uncensored (x0.5 - x1.5)
    "deepseek-v4-pro",
    "grok-4.7",
    "gpt-5.6-terra",
    "claude-haiku-4.5",
    # High-Efficiency Bulk & Utility (x0.05 - x0.38)
    "gpt-6-luna",
    "deepseek-v4.1-flash",
    "deepseek-v4-flash",
    "glm-5.3-flash",
    "minimax-m3",
    "mimo-v2.5",
]

_NON_CHAPTER_NAME_RE = re.compile(
    r"(nav|toc|ncx|cover|titlepage|title_page|copyright|colophon|dedication)",
    re.IGNORECASE,
)
_NAME_RE = re.compile(r"\b[A-Z][A-Za-z'’]{2,}\b")
_SENT_SPLIT = re.compile(r"(?<=[\.!?…])\s+")


# ==============================================================================
# 1. ИСКЛЮЧЕНИЯ И СТРУКТУРЫ ДАННЫХ
# ==============================================================================

class AuthError(Exception):
    pass


class ContentFilterError(Exception):
    pass


class TruncationError(Exception):
    def __init__(self, partial_text: str = ""):
        super().__init__("Model output hit the token cap")
        self.partial_text = partial_text or ""


class TransientNet(Exception):
    pass


class ModelMissing(Exception):
    pass


class AdaptablePayload(Exception):
    def __init__(self, kind: str):
        super().__init__(kind)
        self.kind = kind


class ChapterTriggerError(Exception):
    """Диагностическое исключение для точного отлова триггеров цензуры в GUI."""
    def __init__(self, chapter_idx: int, error_type: str, model: str, snippet: str, raw_error: str):
        self.chapter_idx = chapter_idx
        self.error_type = error_type
        self.model = model
        self.snippet = snippet
        self.raw_error = raw_error
        super().__init__(f"Chapter {chapter_idx} halted by {error_type} in {model}")


class NetState(str, Enum):
    DISPATCH = "dispatch"
    PARSE = "parse"
    CLASSIFY = "classify"
    ADAPT = "adapt"
    RETRY = "retry"


@dataclass
class UniverseProfile:
    work_type: str = "Original Work / Fanfiction"
    fandom: str = "Unknown"
    setting_details: str = ""
    main_characters: Dict[str, str] = field(default_factory=dict)
    key_lore_rules: List[str] = field(default_factory=list)


@dataclass
class LoreEntry:
    canonical_ru: str
    surface_ru: str = ""
    gender: str = "None"
    animacy: str = "inanimate"
    category: str = "concept"
    notes: str = ""

    def use_form(self) -> str:
        return (self.surface_ru or self.canonical_ru or "").strip()


@dataclass
class ChapterSource:
    title: str
    text: str
    spine_id: str = ""


@dataclass
class LLMResult:
    text: str
    elapsed: float
    model: str
    protocol: str


@dataclass(frozen=True)
class EndpointConfig:
    base_url: str = "https://cerberus-api.ddns.net/v1"
    api_key: str = ""
    model: str = "claude-sonnet-5"
    protocol: str = "auto"

    def resolved_protocol(self) -> str:
        return infer_protocol(self.model, self.protocol)


@dataclass(frozen=True)
class PipelineConfig:
    cache_dir: Path = Path("./translation_cache")
    request_timeout: float = 180.0
    max_network_retries: int = 4
    backoff_factor: float = 1.5
    max_concurrent_translations: int = 2
    translation_max_tokens: int = 6144
    harvester_model: str = "deepseek-v4.1-flash"
    arbiter_model: str = "gpt-6-luna"
    reuse_existing_lore: bool = True
    curator: EndpointConfig = field(default_factory=EndpointConfig)
    translator: EndpointConfig = field(default_factory=EndpointConfig)


# ==============================================================================
# 2. БАЗОВЫЕ УТИЛИТЫ И ЛОКАЛЬНЫЙ СЛОВАРЬ (370K СЛОВ)
# ==============================================================================

def infer_protocol(model: str, explicit: str = "") -> str:
    exp = (explicit or "").strip().lower()
    if exp in ("anthropic", "openai"):
        return exp
    name = (model or "").lower()
    if "claude" in name or name.startswith("anthropic"):
        return "anthropic"
    return "openai"


def segment_fingerprint(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


def strip_possessive(token: str) -> str:
    if token.endswith(("'s", "’s")):
        return token[:-2]
    if token.endswith(("'", "’")):
        return token[:-1]
    return token


def normalize_typography(text: str) -> str:
    text = text.replace("\xa0", " ").replace("\u202f", " ").replace("\u200b", "")
    text = text.replace("’", "'").replace("‘", "'").replace("‚", "'")
    text = text.replace("“", '"').replace("”", '"')
    return re.sub(r"[ \t]+", " ", text)


def tail_sentences(text: str, n: int = 2) -> str:
    parts = [p.strip() for p in _SENT_SPLIT.split(text.strip()) if p.strip()]
    if not parts:
        return text[-500:]
    return " ".join(parts[-n:])


def levenshtein_at_most(a: str, b: str, limit: int) -> int:
    if abs(len(a) - len(b)) > limit:
        return limit + 1
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        row_min = i
        for j, cb in enumerate(b, 1):
            cur.append(min(cur[-1] + 1, prev[j] + 1, prev[j - 1] + (ca != cb)))
            row_min = min(row_min, cur[-1])
        if row_min > limit:
            return limit + 1
        prev = cur
    return prev[-1]


class EnglishLexicon:
    _words: Optional[Set[str]] = None

    @classmethod
    def get_lexicon(cls, cache_dir: Path) -> Set[str]:
        if cls._words is not None:
            return cls._words
        cache_dir.mkdir(parents=True, exist_ok=True)
        dict_path = cache_dir / "words_alpha.txt"
        if not dict_path.exists():
            logger.info("Downloading 370k English lexicon (words_alpha.txt)...")
            try:
                urllib.request.urlretrieve(WORDS_ALPHA_URL, dict_path)
            except Exception as e:
                logger.warning("Lexicon download failed (%s); using core fallback.", e)
                cls._words = {
                    "blade", "dragon", "knight", "queen", "king", "christmas", "sword",
                    "red", "rose", "summer", "winter", "autumn", "spring", "death", "fate",
                    "fang", "prince", "earth", "festival", "market", "white", "black",
                }
                return cls._words
        try:
            with open(dict_path, "r", encoding="utf-8") as f:
                cls._words = {line.strip().lower() for line in f if line.strip()}
            logger.info("English lexicon ready: %d words in RAM", len(cls._words))
        except Exception as e:
            logger.warning("Lexicon parse failed: %s", e)
            cls._words = set()
        return cls._words


# ==============================================================================
# 3. КОНТРОЛЛЕР AIMD, МЕТРИКИ И ЗАЩИТНЫЕ ПРАВИЛА
# ==============================================================================

@dataclass
class AdaptiveController:
    max_workers: int = 2
    target_min_s: float = 15.0
    target_max_s: float = 45.0
    lore_batch_size: int = 16
    min_lore_batch: int = 6
    max_lore_batch: int = 28
    chunk_words: int = 1100
    min_chunk_words: int = 750
    max_chunk_words: int = 1800
    measured_tok_s: float = 40.0
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    def calibrate(self, measured_tok_s: float) -> None:
        self.measured_tok_s = max(15.0, measured_tok_s)
        calculated = int((self.measured_tok_s * 35.0) / 1.4)
        self.chunk_words = max(self.min_chunk_words, min(self.max_chunk_words, calculated))
        logger.info("[AIMD] %.1f tok/s | chunk %d words | lore batch %d",
                    self.measured_tok_s, self.chunk_words, self.lore_batch_size)

    def snapshot_chunk(self) -> int:
        return self.chunk_words

    async def report_latency(self, elapsed: float, stage: str = "lore") -> None:
        async with self._lock:
            if elapsed > self.target_max_s:
                if stage == "lore":
                    self.lore_batch_size = max(self.min_lore_batch, int(self.lore_batch_size * 0.8))
                elif stage == "trans":
                    self.chunk_words = max(self.min_chunk_words, int(self.chunk_words * 0.9))
            elif elapsed < self.target_min_s:
                if stage == "lore" and self.lore_batch_size < self.max_lore_batch:
                    self.lore_batch_size += 1
                elif stage == "trans" and self.chunk_words < self.max_chunk_words:
                    self.chunk_words = min(self.max_chunk_words, self.chunk_words + 80)

    async def report_network_drop(self, stage: str = "lore") -> None:
        """Вызывается ТОЛЬКО при реальных сетевых сбоях (TransientNet), а не при логических ошибках."""
        async with self._lock:
            if stage == "lore":
                self.lore_batch_size = max(self.min_lore_batch, int(self.lore_batch_size * 0.7))
            elif stage == "trans":
                self.chunk_words = max(self.min_chunk_words, int(self.chunk_words * 0.8))
            logger.warning("[AIMD] Network drop detected, %s window reduced", stage)


@dataclass
class StatsTracker:
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    total_words_original: int = 0
    start_time: float = field(default_factory=time.time)
    translate_start_time: Optional[float] = None
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    async def add_usage(self, prompt_tokens: int, completion_tokens: int) -> None:
        async with self._lock:
            self.total_prompt_tokens += prompt_tokens
            self.total_completion_tokens += completion_tokens

    def mark_translate_start(self) -> None:
        if self.translate_start_time is None:
            self.translate_start_time = time.time()

    def get_summary(self) -> str:
        elapsed = max(1.0, time.time() - self.start_time)
        mins, secs = int(elapsed // 60), int(elapsed % 60)
        total_tokens = self.total_prompt_tokens + self.total_completion_tokens
        speed_elapsed = max(1.0, time.time() - (self.translate_start_time or self.start_time))
        wpm = int((self.total_words_original / speed_elapsed) * 60) if self.total_words_original else 0
        return (
            f"Time {mins}m {secs}s | Tokens {total_tokens:,} "
            f"(in {self.total_prompt_tokens:,} / out {self.total_completion_tokens:,}) | "
            f"~{wpm:,} wpm"
        )


class Guardrails:
    _REFUSAL_REGEX = re.compile(
        r"(i cannot fulfill|i am unable to|as an ai language model|"
        r"my safety policies|i must decline|violates safety guidelines|content policy|"
        r"无法给到相关内容|不符合安全规范)",
        re.IGNORECASE,
    )
    _MUTANT_VERBS_REGEX = re.compile(
        r"\b[а-яА-ЯёЁ]{3,}(?:аллка|овалка|стоналлка|ложилка|получилка|деформовалка)\b",
        re.IGNORECASE,
    )
    _ZERO_PUNCTUATION_LOOP_REGEX = re.compile(
        r"(?:\b[а-яА-ЯёЁa-zA-Z0-9-]+\b\s+){45,}[а-яА-ЯёЁa-zA-Z0-9-]+"
    )
    _ENGLISH_LEAK_REGEX = re.compile(r"(?:\b[a-zA-Z]+\b[\s,;]+){25,}")

    @classmethod
    def is_refusal(cls, text: Optional[str]) -> bool:
        if not text:
            return True
        return bool(cls._REFUSAL_REGEX.search(text[:300]))

    @classmethod
    def is_degenerated(cls, text: str) -> bool:
        if not text or len(text) < 60:
            return False
        if cls._MUTANT_VERBS_REGEX.search(text):
            return True
        if cls._ZERO_PUNCTUATION_LOOP_REGEX.search(text):
            return True
        if cls._ENGLISH_LEAK_REGEX.search(text):
            return True
        if re.search(r"(.{35,})\1{4,}", text):
            return True
        return False

    @staticmethod
    def extract_json_block(raw_text: str) -> str:
        if not raw_text:
            return ""

        def _extract(open_ch: str, close_ch: str) -> str:
            start = raw_text.find(open_ch)
            if start == -1:
                return ""
            depth = 0
            in_str = False
            escape = False
            for i in range(start, len(raw_text)):
                ch = raw_text[i]
                if in_str:
                    if escape:
                        escape = False
                    elif ch == "\\":
                        escape = True
                    elif ch == '"':
                        in_str = False
                    continue
                if ch == '"':
                    in_str = True
                    continue
                if ch == open_ch:
                    depth += 1
                elif ch == close_ch:
                    depth -= 1
                    if depth == 0:
                        return raw_text[start : i + 1]
            return raw_text[start:]

        obj = _extract("{", "}")
        arr = _extract("[", "]")
        if obj and arr:
            return obj if raw_text.find("{") <= raw_text.find("[") else arr
        return obj or arr or raw_text


# ==============================================================================
# 4. МЕНЕДЖЕР КЭША (С ПОДДЕРЖКОЙ ПЕРЕИСПОЛЬЗОВАНИЯ MASTER LORE)
# ==============================================================================

class CacheManager:
    def __init__(self, cache_root: Path, book_id: str, translator_model: str):
        self.book_cache_dir = cache_root / book_id
        self.book_cache_dir.mkdir(parents=True, exist_ok=True)
        self.translator_model = translator_model

    def _meta_path(self) -> Path:
        return self.book_cache_dir / "master_lore.json"

    def _preprocessed_path(self) -> Path:
        return self.book_cache_dir / "preprocessed_chapters.json"

    def _text_path(self, idx: int) -> Path:
        return self.book_cache_dir / f"{idx:03d}.json"

    def _partial_path(self, idx: int) -> Path:
        return self.book_cache_dir / f"{idx:03d}.partial.json"

    @staticmethod
    def _atomic_write_text(path: Path, content: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=str(path.parent))
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(content)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_name, path)
        except Exception:
            try:
                os.unlink(tmp_name)
            except OSError:
                pass
            raise

    def invalidate_chapters(self, chapter_indices: Set[int]) -> None:
        for idx in chapter_indices:
            for p in (self._text_path(idx), self._partial_path(idx)):
                if p.exists():
                    p.unlink()

    def load_preprocessed_chapters(self, expected_count: int) -> Optional[List[ChapterSource]]:
        path = self._preprocessed_path()
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(data, list) or len(data) != expected_count:
                return None
            return [
                ChapterSource(title=item["title"], text=item["text"], spine_id=item.get("spine_id", ""))
                for item in data
            ]
        except Exception:
            return None

    def save_preprocessed_chapters(self, chapters: List[ChapterSource]) -> None:
        payload = [{"title": c.title, "text": c.text, "spine_id": c.spine_id} for c in chapters]
        self._atomic_write_text(self._preprocessed_path(), json.dumps(payload, ensure_ascii=False))

    def load_master_lore(self, expected_model: str, allow_legacy: bool = True) -> Optional[Tuple[UniverseProfile, Dict[str, LoreEntry]]]:
        path = self._meta_path()
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if not allow_legacy:
                if int(data.get("schema_version", 0) or 0) != LORE_SCHEMA_VERSION:
                    return None
                if data.get("curator_model") != expected_model:
                    return None
            profile_raw = data.get("profile", {})
            profile = UniverseProfile(
                work_type=profile_raw.get("work_type", "Original Work / Fanfiction"),
                fandom=profile_raw.get("fandom", "Unknown"),
                setting_details=profile_raw.get("setting_details", ""),
                main_characters=profile_raw.get("main_characters", {}),
                key_lore_rules=profile_raw.get("key_lore_rules", []),
            )
            glossary = {}
            for k, v in data.get("glossary", {}).items():
                key_clean = k.strip()
                if not key_clean:
                    continue
                canonical = v.get("canonical_ru", key_clean)
                surface = v.get("surface_ru") or canonical
                glossary[key_clean] = LoreEntry(
                    canonical_ru=canonical,
                    surface_ru=surface,
                    gender=v.get("gender", "None"),
                    animacy=v.get("animacy", "inanimate"),
                    category=v.get("category", "concept"),
                    notes=v.get("notes", ""),
                )
            return profile, glossary
        except Exception as e:
            logger.warning("Lore cache unreadable: %s", e)
            return None

    def save_master_lore(self, profile: UniverseProfile, glossary: Dict[str, LoreEntry], curator_model: str) -> None:
        payload = {
            "schema_version": LORE_SCHEMA_VERSION,
            "prompt_version": PROMPT_VERSION,
            "curator_model": curator_model,
            "profile": {
                "work_type": profile.work_type,
                "fandom": profile.fandom,
                "setting_details": profile.setting_details,
                "main_characters": profile.main_characters,
                "key_lore_rules": profile.key_lore_rules,
            },
            "glossary": {
                k: {
                    "canonical_ru": v.canonical_ru,
                    "surface_ru": v.surface_ru,
                    "gender": v.gender,
                    "animacy": v.animacy,
                    "category": v.category,
                    "notes": v.notes,
                }
                for k, v in glossary.items()
            },
        }
        self._atomic_write_text(self._meta_path(), json.dumps(payload, ensure_ascii=False, indent=2))

    def load_translation(self, idx: int, source_sha: str) -> Optional[str]:
        path = self._text_path(idx)
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if data.get("model") != self.translator_model:
                return None
            if data.get("source_sha") and data.get("source_sha") != source_sha:
                return None
            text = (data.get("text") or "").strip()
            if not text or Guardrails.is_refusal(text) or Guardrails.is_degenerated(text):
                return None
            return text
        except Exception:
            return None

    def save_translation(self, idx: int, content: str, model: str, source_sha: str) -> None:
        payload = {
            "prompt_version": PROMPT_VERSION,
            "model": model,
            "source_sha": source_sha,
            "text": content,
        }
        self._atomic_write_text(self._text_path(idx), json.dumps(payload, ensure_ascii=False))
        self._partial_path(idx).unlink(missing_ok=True)

    def load_partial_segments(self, idx: int, source_segments: List[str]) -> List[str]:
        path = self._partial_path(idx)
        if not path.exists():
            return []
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if data.get("model") != self.translator_model:
                return []
            saved_fps = data.get("fingerprints") or []
            segs = data.get("segments") or []
            current_fps = [segment_fingerprint(s) for s in source_segments]
            clean: List[str] = []
            for i, s in enumerate(segs):
                if not isinstance(s, str) or not s.strip():
                    break
                if Guardrails.is_refusal(s) or Guardrails.is_degenerated(s):
                    break
                if i >= len(current_fps) or (i < len(saved_fps) and saved_fps[i] != current_fps[i]):
                    break
                clean.append(s)
            return clean
        except Exception:
            return []

    def save_partial_segments(self, idx: int, segments: List[str], source_segments: List[str], model: str) -> None:
        payload = {
            "prompt_version": PROMPT_VERSION,
            "model": model,
            "fingerprints": [segment_fingerprint(s) for s in source_segments[: len(segments)]],
            "segments": segments,
        }
        try:
            self._atomic_write_text(self._partial_path(idx), json.dumps(payload, ensure_ascii=False))
        except Exception as e:
            logger.warning("Partial cache write failed for chapter %d: %s", idx, e)


# ==============================================================================
# 5. СЕТЕВОЙ ДВИЖОК (FSM + DUAL PROTOCOL)
# ==============================================================================

class DualProtocolLLMClient:
    CENSORSHIP_SIGNATURES = (
        "cyber flagged", "upstream_content", "content_policy", "violates safety",
        "safety_policy", "blocked by content filter", "flagged",
    )

    def __init__(self, endpoint_cfg: EndpointConfig, timeout: float, max_retries: int, backoff: float, stats: StatsTracker):
        self.endpoint_cfg = endpoint_cfg
        self.timeout = timeout
        self.max_retries = max_retries
        self.backoff_factor = backoff
        self.stats = stats
        self.client_timeout = httpx.Timeout(self.timeout, connect=20.0, read=self.timeout, write=30.0)
        self._client: Optional[httpx.AsyncClient] = None

    def _http(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(timeout=self.client_timeout)
        return self._client

    async def aclose(self) -> None:
        if self._client is not None and not self._client.is_closed:
            await self._client.aclose()
        self._client = None

    @classmethod
    async def fetch_available_models(cls, base_url: str, api_key: str) -> List[str]:
        url = base_url.rstrip("/") + "/models"
        headers = {}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
            headers["x-api-key"] = api_key
        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                res = await client.get(url, headers=headers)
                if res.status_code == 200:
                    data = res.json()
                    return sorted(m["id"] for m in data.get("data", []) if isinstance(m, dict) and "id" in m)
        except Exception as e:
            logger.warning("GET /models failed: %s", e)
        return []

    @staticmethod
    async def _read_error_body(response: httpx.Response) -> str:
        try:
            raw = await response.aread()
            return raw.decode("utf-8", errors="ignore")[:400]
        except Exception:
            return ""

    @staticmethod
    async def _iter_sse_events(response: httpx.Response) -> AsyncIterator[Dict[str, Any]]:
        async for line in response.aiter_lines():
            line = line.strip()
            if not line.startswith("data:"):
                continue
            data_str = line[5:].strip()
            if not data_str or data_str == "[DONE]":
                if data_str == "[DONE]":
                    return
                continue
            try:
                yield json.loads(data_str)
            except json.JSONDecodeError:
                continue

    def _classify_status(self, status: int, body: str) -> Exception:
        low = (body or "").lower()
        if status in (401, 403):
            return AuthError(f"HTTP {status}: {body[:180]}")
        if any(sig in low for sig in self.CENSORSHIP_SIGNATURES):
            return ContentFilterError(f"HTTP {status}: {body[:180]}")
        if status == 400 and "thinking" in low:
            return AdaptablePayload("thinking")
        if status == 400 and "stream_options" in low:
            return AdaptablePayload("stream_options")
        if status == 400 and ("response_format" in low or "json_object" in low):
            return AdaptablePayload("json_mode")
        if status == 404 or "model_not_found" in low or "no such model" in low or "does not exist" in low:
            return ModelMissing(body[:180])
        return TransientNet(f"HTTP {status}: {body[:180]}")

    async def run_reference_probe(self) -> float:
        logger.info("[probe] %s (%s)", self.endpoint_cfg.base_url, self.endpoint_cfg.model)
        t0 = time.perf_counter()
        try:
            result = await self.call(
                messages=[{
                    "role": "user",
                    "content": "Translate into Russian in one short sentence: "
                               "'The cold autumn wind moved silently over the empty harbor.'",
                }],
                temperature=0.2,
                max_tokens=150,
                disable_thinking=True,
                stream=True,
            )
            elapsed = max(0.5, time.perf_counter() - t0)
            approx_out = max(8, len((result.text or "").split()))
            return max(15.0, (approx_out * 1.3) / elapsed)
        except Exception as e:
            logger.warning("Probe failed (%s); default 40 tok/s", e)
            return 40.0

    async def call(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.3,
        max_tokens: int = 4096,
        is_json: bool = False,
        disable_thinking: bool = False,
        stream: Optional[bool] = None,
        model_override: Optional[str] = None,
    ) -> LLMResult:
        use_stream = (not is_json) if stream is None else stream
        t0 = time.perf_counter()
        model = model_override or self.endpoint_cfg.model
        protocol = infer_protocol(model, self.endpoint_cfg.protocol if not model_override else "")
        text = await self._endpoint(
            model, protocol, messages, temperature, max_tokens, is_json, disable_thinking, use_stream,
        )
        return LLMResult(text, time.perf_counter() - t0, model, protocol)

    async def _endpoint(
        self,
        model: str,
        protocol: str,
        messages: List[Dict[str, str]],
        temperature: float,
        max_tokens: int,
        is_json: bool,
        disable_thinking: bool,
        use_stream: bool,
    ) -> str:
        stripped = {"thinking": False, "stream_options": False, "json_mode": False}
        last_error: Optional[Exception] = None
        state = NetState.DISPATCH
        attempt = 1
        while attempt <= self.max_retries:
            if state in (NetState.DISPATCH, NetState.ADAPT, NetState.RETRY):
                try:
                    content, in_tok, out_tok, finish = await self._dispatch(
                        model, protocol, messages, temperature, max_tokens, is_json,
                        disable_thinking, use_stream, stripped,
                    )
                    state = NetState.PARSE
                except AdaptablePayload as exc:
                    stripped[exc.kind] = True
                    state = NetState.ADAPT
                    continue
                except (AuthError, ContentFilterError, ModelMissing):
                    raise
                except Exception as exc:
                    last_error = exc
                    state = NetState.RETRY
                    await self._sleep_retry(attempt, str(exc))
                    attempt += 1
                    continue

            state = NetState.CLASSIFY
            if not content:
                last_error = TransientNet("empty response")
                state = NetState.RETRY
                await self._sleep_retry(attempt, "empty")
                attempt += 1
                continue
            if finish in ("max_tokens", "length"):
                raise TruncationError(content)
            if Guardrails.is_refusal(content):
                raise ContentFilterError(f"Model refusal: {content[:160]}")
            if not is_json and Guardrails.is_degenerated(content):
                last_error = TransientNet("degenerate output")
                state = NetState.RETRY
                await self._sleep_retry(attempt, "degenerate")
                attempt += 1
                continue
            await self.stats.add_usage(in_tok, out_tok)
            return content
        raise TransientNet(f"{model} exhausted network retries: {last_error}")

    async def _sleep_retry(self, attempt: int, reason: str) -> None:
        low = reason.lower()
        sleep_time = random.uniform(22.0, 32.0) if "429" in low else (self.backoff_factor ** attempt) + random.uniform(1.5, 3.0)
        logger.warning("[net] retry %d/%d in %.1fs (%s)", attempt, self.max_retries, sleep_time, reason[:70])
        await asyncio.sleep(sleep_time)

    def _headers(self, protocol: str) -> Dict[str, str]:
        headers = {
            "Authorization": f"Bearer {self.endpoint_cfg.api_key}",
            "x-api-key": self.endpoint_cfg.api_key,
            "Content-Type": "application/json",
        }
        if protocol == "anthropic":
            headers["anthropic-version"] = "2023-06-01"
        return headers

    async def _dispatch(
        self,
        model: str,
        protocol: str,
        messages: List[Dict[str, str]],
        temperature: float,
        max_tokens: int,
        is_json: bool,
        disable_thinking: bool,
        use_stream: bool,
        stripped: Dict[str, bool],
    ) -> Tuple[str, int, int, str]:
        if protocol == "anthropic":
            return await self._dispatch_anthropic(
                model, messages, temperature, max_tokens, is_json, disable_thinking, use_stream, stripped,
            )
        return await self._dispatch_openai(
            model, messages, temperature, max_tokens, is_json, use_stream, stripped,
        )

    async def _dispatch_anthropic(
        self,
        model: str,
        messages: List[Dict[str, str]],
        temperature: float,
        max_tokens: int,
        is_json: bool,
        disable_thinking: bool,
        use_stream: bool,
        stripped: Dict[str, bool],
    ) -> Tuple[str, int, int, str]:
        url = self.endpoint_cfg.base_url.rstrip("/") + "/messages"
        system_text = ""
        user_msgs = []
        for m in messages:
            if m["role"] == "system":
                system_text += m["content"] + "\n\n"
            else:
                user_msgs.append({"role": m["role"], "content": m["content"]})
        payload: Dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "temperature": 0.1 if is_json else temperature,
            "messages": user_msgs,
            "stream": use_stream,
        }
        if system_text.strip():
            payload["system"] = system_text.strip()
        if disable_thinking and not stripped["thinking"] and "claude" in model.lower():
            payload["thinking"] = {"type": "disabled"}
        headers = self._headers("anthropic")
        if use_stream:
            return await self._anthropic_stream(url, headers, payload)
        return await self._anthropic_buffered(url, headers, payload)

    async def _dispatch_openai(
        self,
        model: str,
        messages: List[Dict[str, str]],
        temperature: float,
        max_tokens: int,
        is_json: bool,
        use_stream: bool,
        stripped: Dict[str, bool],
    ) -> Tuple[str, int, int, str]:
        url = self.endpoint_cfg.base_url.rstrip("/") + "/chat/completions"
        payload: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": 0.2 if is_json else temperature,
            "max_tokens": max_tokens,
            "stream": use_stream,
        }
        if is_json and not stripped["json_mode"]:
            payload["response_format"] = {"type": "json_object"}
        if use_stream and not stripped["stream_options"]:
            payload["stream_options"] = {"include_usage": True}
        headers = self._headers("openai")
        if use_stream:
            return await self._openai_stream(url, headers, payload)
        return await self._openai_buffered(url, headers, payload)

    async def _anthropic_buffered(self, url: str, headers: Dict[str, str], payload: Dict[str, Any]) -> Tuple[str, int, int, str]:
        res = await self._http().post(url, headers=headers, json=payload)
        if res.status_code != 200:
            raise self._classify_status(res.status_code, res.text[:400])
        data = res.json()
        blocks = data.get("content", [])
        text = "".join(b.get("text", "") for b in blocks if isinstance(b, dict) and b.get("type") == "text").strip()
        usage = data.get("usage", {})
        return text, int(usage.get("input_tokens", 0) or 0), int(usage.get("output_tokens", 0) or 0), str(data.get("stop_reason") or "")

    async def _anthropic_stream(self, url: str, headers: Dict[str, str], payload: Dict[str, Any]) -> Tuple[str, int, int, str]:
        text_parts: List[str] = []
        stop_reason = ""
        in_tok = 0
        out_tok = 0
        try:
            async with self._http().stream("POST", url, headers=headers, json=payload) as response:
                if response.status_code != 200:
                    body = await self._read_error_body(response)
                    raise self._classify_status(response.status_code, body)
                async for event in self._iter_sse_events(response):
                    ev_type = event.get("type")
                    if ev_type == "error":
                        raise self._classify_status(400, str(event.get("error") or event))
                    if ev_type == "content_block_delta":
                        delta = event.get("delta") or {}
                        if delta.get("type") == "text_delta":
                            text_parts.append(delta.get("text") or "")
                    elif ev_type == "content_block_start":
                        block = event.get("content_block") or {}
                        if block.get("type") == "text" and block.get("text"):
                            text_parts.append(block["text"])
                    elif ev_type == "message_start":
                        in_tok = int((event.get("message") or {}).get("usage", {}).get("input_tokens", 0) or 0)
                    elif ev_type == "message_delta":
                        out_tok = int((event.get("usage") or {}).get("output_tokens", 0) or 0)
                        stop_reason = str((event.get("delta") or {}).get("stop_reason") or stop_reason)
                    elif ev_type == "message_stop":
                        break
        except (AuthError, ContentFilterError, AdaptablePayload, ModelMissing, TransientNet):
            raise
        except Exception as exc:
            raise TransientNet(f"stream dropped: {exc}") from exc
        content = "".join(text_parts).strip()
        if not in_tok:
            in_tok = max(10, int(len(json.dumps(payload.get("messages", []), ensure_ascii=False)) / 3.2))
        if not out_tok:
            out_tok = max(10, int(len(content) / 3.2))
        return content, in_tok, out_tok, stop_reason

    async def _openai_buffered(self, url: str, headers: Dict[str, str], payload: Dict[str, Any]) -> Tuple[str, int, int, str]:
        res = await self._http().post(url, headers=headers, json={**payload, "stream": False})
        if res.status_code != 200:
            raise self._classify_status(res.status_code, res.text[:400])
        data = res.json()
        choice = (data.get("choices") or [{}])[0]
        message = choice.get("message") or {}
        content = (message.get("content") or "").strip()
        usage = data.get("usage") or {}
        return content, int(usage.get("prompt_tokens", 0) or 0), int(usage.get("completion_tokens", 0) or 0), str(choice.get("finish_reason") or "").lower()

    async def _openai_stream(self, url: str, headers: Dict[str, str], payload: Dict[str, Any]) -> Tuple[str, int, int, str]:
        text_parts: List[str] = []
        finish = ""
        in_tok = 0
        out_tok = 0
        try:
            async with self._http().stream("POST", url, headers=headers, json=payload) as response:
                if response.status_code != 200:
                    body = await self._read_error_body(response)
                    raise self._classify_status(response.status_code, body)
                async for event in self._iter_sse_events(response):
                    if event.get("error"):
                        raise self._classify_status(400, str(event["error"]))
                    choices = event.get("choices") or []
                    if choices:
                        delta = choices[0].get("delta") or {}
                        piece = delta.get("content") or ""
                        if piece:
                            text_parts.append(piece)
                        if choices[0].get("finish_reason"):
                            finish = str(choices[0]["finish_reason"]).lower()
                    usage = event.get("usage")
                    if usage:
                        in_tok = int(usage.get("prompt_tokens", 0) or 0)
                        out_tok = int(usage.get("completion_tokens", 0) or 0)
        except (AuthError, ContentFilterError, AdaptablePayload, ModelMissing, TransientNet):
            raise
        except Exception as exc:
            raise TransientNet(f"stream dropped: {exc}") from exc
        content = "".join(text_parts).strip()
        if not in_tok:
            in_tok = max(10, int(len(json.dumps(payload.get("messages", []), ensure_ascii=False)) / 3.2))
        if not out_tok:
            out_tok = max(10, int(len(content) / 3.2))
        return content, in_tok, out_tok, finish


# ==============================================================================
# 6. ФОРМАТИРОВАНИЕ И РАБОТА С EPUB
# ==============================================================================

class TextFormatter:
    _LEAF_BLOCKS = frozenset({"p", "h1", "h2", "h3", "h4", "h5", "h6", "li", "blockquote", "pre"})

    @staticmethod
    def _inline_markdown(elem: Tag) -> str:
        parts: List[str] = []
        for child in elem.children:
            if isinstance(child, NavigableString):
                parts.append(str(child))
            elif isinstance(child, Tag):
                name = child.name.lower()
                if name in ("i", "em"):
                    parts.append(f"*{TextFormatter._inline_markdown(child)}*")
                elif name in ("b", "strong"):
                    parts.append(f"**{TextFormatter._inline_markdown(child)}**")
                elif name == "br":
                    parts.append("\n")
                elif name in TextFormatter._LEAF_BLOCKS:
                    continue
                else:
                    parts.append(TextFormatter._inline_markdown(child))
        return "".join(parts)

    @classmethod
    def html_to_markdown(cls, html_content: str) -> str:
        if isinstance(html_content, (bytes, bytearray)):
            html_content = html_content.decode("utf-8", errors="ignore")
        soup = BeautifulSoup(html_content, "html.parser")
        for junk in soup.find_all(["script", "style"]):
            junk.decompose()
        paragraphs: List[str] = []

        def walk(node: Any) -> None:
            if isinstance(node, NavigableString) or not isinstance(node, Tag):
                return
            name = (node.name or "").lower()
            if name == "hr":
                paragraphs.append("* * *")
                return
            if name in cls._LEAF_BLOCKS:
                text = re.sub(r"[ \t]+", " ", cls._inline_markdown(node).strip())
                if text:
                    paragraphs.append(text)
                return
            for child in node.children:
                walk(child)

        walk(soup.body or soup)
        if not paragraphs:
            return soup.get_text(separator="\n\n").strip()
        return "\n\n".join(paragraphs)

    @staticmethod
    def markdown_to_html(md_text: str) -> str:
        raw_paragraphs = [p.strip() for p in re.split(r"\n\s*\n", md_text) if p.strip()]
        html_parts = []
        for p in raw_paragraphs:
            if p in ["* * *", "***", "---", "___"]:
                html_parts.append('<p style="text-align: center; margin: 1.5em 0;">* * *</p>')
                continue
            p_html = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", p)
            p_html = re.sub(r"\*(.+?)\*", r"<em>\1</em>", p_html)
            p_html = re.sub(r"_(.+?)_", r"<em>\1</em>", p_html)
            html_parts.append(f"<p>{p_html}</p>")
        return "\n".join(html_parts)

    @staticmethod
    def post_lint(text: str) -> str:
        text = re.sub(r"[ \t]{2,}", " ", text)
        text = re.sub(r"(?m)^[ \t]*[\-\–]\s+", "— ", text)
        text = re.sub(r"\s+([,;:\.!\?])", r"\1", text)
        return text.strip()

    @staticmethod
    def guess_chapter_title(md_text: str, fallback: str) -> str:
        for line in md_text.splitlines():
            s = line.strip().lstrip("#").strip()
            if not s:
                continue
            if 1 <= len(s) <= 80 and len(s.split()) <= 12:
                return s
            break
        return fallback


class EpubService:
    EPUB_CSS = """
    body { font-family: serif; margin: 5%; text-align: justify; line-height: 1.45; }
    h2 { text-align: center; margin-top: 1em; margin-bottom: 1.5em; }
    p { text-indent: 1.5em; margin: 0; margin-bottom: 0.3em; }
    em { font-style: italic; }
    strong { font-weight: bold; }
    """

    @classmethod
    def extract_chapters_and_metadata(cls, epub_path: Path, min_word_count: int = 40) -> Tuple[str, List[ChapterSource]]:
        book = epub.read_epub(str(epub_path))
        title_meta = book.get_metadata("DC", "title")
        book_title = title_meta[0][0] if title_meta else epub_path.stem
        chapters: List[ChapterSource] = []
        seen_ids: Set[str] = set()
        spine_entries = list(book.spine or [])
        if not spine_entries:
            spine_entries = [
                (item.get_id(), "yes")
                for item in book.get_items()
                if item.get_type() == ebooklib.ITEM_DOCUMENT
            ]
        for entry in spine_entries:
            item_id = entry[0] if isinstance(entry, (tuple, list)) else entry
            if not item_id or item_id == "nav":
                continue
            item = book.get_item_with_id(item_id)
            if item is None or item.get_type() != ebooklib.ITEM_DOCUMENT or item_id in seen_ids:
                continue
            seen_ids.add(item_id)
            fname = (getattr(item, "file_name", None) or item_id or "").lower()
            if _NON_CHAPTER_NAME_RE.search(fname) or _NON_CHAPTER_NAME_RE.search(str(item_id)):
                continue
            try:
                body = item.get_body_content()
            except Exception:
                continue
            md_text = TextFormatter.html_to_markdown(body)
            words = md_text.split()
            if len(words) < min_word_count:
                continue
            is_first = len(chapters) == 0
            is_prologue = is_first and (
                any(k in md_text[:300].lower() for k in ["prologue", "пролог"])
                or len(words) < 500
            )
            fallback = "Пролог" if is_prologue else f"Глава {len([c for c in chapters if c.title.lower() not in ('пролог', 'prologue')]) + 1}"
            chapters.append(ChapterSource(
                title=TextFormatter.guess_chapter_title(md_text, fallback),
                text=normalize_typography(md_text),
                spine_id=str(item_id),
            ))
        return book_title, chapters

    @classmethod
    def build_epub(cls, chapters: List[ChapterSource], output_path: Path, title: str) -> None:
        book = epub.EpubBook()
        book.set_identifier("ai-literary-pipeline-v10.0")
        book.set_title(title)
        book.set_language("ru")
        book.add_author("AI Literary Translator")
        epub_chapters = []
        spine_items = []
        for idx, chapter in enumerate(chapters, 1):
            heading = chapter.title or f"Глава {idx}"
            full_html = f"<h2>{html.escape(heading)}</h2>\n" + TextFormatter.markdown_to_html(chapter.text)
            c = epub.EpubHtml(title=heading, file_name=f"chapter_{idx:03d}.xhtml", lang="ru")
            c.content = full_html
            book.add_item(c)
            epub_chapters.append(c)
            spine_items.append(c)
        book.toc = tuple(epub_chapters)
        book.add_item(epub.EpubNcx())
        book.add_item(epub.EpubNav())
        book.add_item(epub.EpubItem(
            uid="style_nav",
            file_name="style/nav.css",
            media_type="text/css",
            content=cls.EPUB_CSS,
        ))
        book.spine = ["nav"] + spine_items
        epub.write_epub(str(output_path), book)


# ==============================================================================
# 7. ПРЕПРОЦЕССИНГ: LLM-ЧИСТКА АННОТАЦИЙ И ДВУХУРОВНЕВЫЙ НОРМАЛИЗАТОР
# ==============================================================================

class LLMAnnotationCleaner:
    """
    Чистка авторских ремарок (A/N, дисклеймеров, благодарностей) в начале и конце глав
    без регулярок — через поиск точной цитаты (Anchor Substring) с помощью дешевой LLM.
    """
    @classmethod
    async def clean_chapters(
        cls,
        chapters: List[ChapterSource],
        llm_caller: Callable[..., Any],
        model: str = "deepseek-v4.1-flash",
        batch_size: int = 15,
    ) -> None:
        logger.info("[0.5/3] Scanning chapter openings/endings for Author Notes via %s...", model)
        system_prompt = (
            "You are a pre-processing text analyzer for a novel.\n"
            "For each chapter sample, check if the opening has an Author's Note (A/N, disclaimer, beta thanks, Patreon link) "
            "BEFORE the story prose starts, or if the ending has an Author's Note AFTER the story prose ends.\n\n"
            "Rules:\n"
            "1. If the opening starts directly with story narrative -> set has_header_note to false.\n"
            "2. If the opening HAS an author note -> set has_header_note to true, and provide 'story_start_anchor': "
            "an EXACT verbatim quote of 6-10 words where the actual story prose begins.\n"
            "3. If the ending has a trailing author note -> set has_footer_note to true, and provide 'footer_note_anchor': "
            "an EXACT verbatim quote of 6-10 words where the trailing author note begins.\n\n"
            "Return JSON format:\n"
            "{\n"
            '  "1": {"has_header_note": false, "has_footer_note": false},\n'
            '  "2": {"has_header_note": true, "story_start_anchor": "The cold rain beat against the window", "has_footer_note": false}\n'
            "}"
        )

        stripped_count = 0
        for i in range(0, len(chapters), batch_size):
            chunk = chapters[i : i + batch_size]
            payload = {}
            for offset, ch in enumerate(chunk):
                idx_str = str(i + offset + 1)
                head = ch.text[:900].strip()
                tail = ch.text[-900:].strip() if len(ch.text) > 1800 else ""
                payload[idx_str] = {"opening": head, "ending": tail}

            try:
                result = await llm_caller(
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
                    ],
                    temperature=0.1,
                    max_tokens=1500,
                    is_json=True,
                    disable_thinking=True,
                    stream=False,
                    model_override=model,
                )
                data = json_repair.loads(Guardrails.extract_json_block(result.text))
                if not isinstance(data, dict):
                    continue

                for offset, ch in enumerate(chunk):
                    idx_str = str(i + offset + 1)
                    info = data.get(idx_str)
                    if not isinstance(info, dict):
                        continue

                    if info.get("has_header_note") and info.get("story_start_anchor"):
                        anchor = str(info["story_start_anchor"]).strip()
                        pos = ch.text.find(anchor)
                        if 0 < pos < 2500:
                            ch.text = ch.text[pos:].lstrip()
                            stripped_count += 1

                    if info.get("has_footer_note") and info.get("footer_note_anchor"):
                        anchor = str(info["footer_note_anchor"]).strip()
                        pos = ch.text.rfind(anchor)
                        if pos > len(ch.text) - 2500 and pos > 200:
                            ch.text = ch.text[:pos].rstrip()
                            stripped_count += 1
            except Exception as e:
                logger.warning("Annotation scan batch %d skipped (%s)", i // batch_size + 1, e)

        logger.info("Annotation cleaner finished: %d author note blocks sliced off.", stripped_count)


class SourceTypoNormalizer:
    """
    Двухуровневый нормализатор источника:
    Tier 1: Локальный словарь 370k слов отсекает валидные английские слова (Blade, Dragon, Christmas).
    Tier 2: Бинарный арбитр на gpt-6-luna проверяет редкие опечатки имен.
    """
    def __init__(
        self,
        master_glossary: Dict[str, LoreEntry],
        chapters: List[ChapterSource],
        cache_dir: Path,
        forced: Optional[Dict[str, str]] = None,
    ):
        self.glossary = master_glossary
        self.chapters = chapters
        self.lexicon = EnglishLexicon.get_lexicon(cache_dir)
        self.forced = {k.strip(): v.strip() for k, v in (forced or {}).items() if k.strip() and v.strip()}
        self.replacements: Dict[str, str] = {}

    def _collect_canons(self) -> Dict[str, str]:
        canons = {}
        for key in self.glossary.keys():
            k = key.strip()
            if k and " " not in k and "-" not in k and k.isalpha():
                canons[k.lower()] = k
        return canons

    def _frequencies(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for ch in self.chapters:
            for match in _NAME_RE.finditer(ch.text):
                low = strip_possessive(match.group(0)).lower()
                counts[low] = counts.get(low, 0) + 1
        return counts

    def harvest_oov_candidates(self) -> List[Dict[str, str]]:
        canons = self._collect_canons()
        by_len: Dict[int, List[Tuple[str, str]]] = {}
        for c_low, c_orig in canons.items():
            by_len.setdefault(len(c_low), []).append((c_low, c_orig))
        freq = self._frequencies()
        candidates: List[Dict[str, str]] = []
        seen = set()
        for ch in self.chapters:
            for match in _NAME_RE.finditer(ch.text):
                token = strip_possessive(match.group(0))
                low = token.lower()
                if low in seen or len(low) <= 4 or low in self.lexicon or low in canons:
                    continue
                if freq.get(low, 0) > TYPO_MAX_HITS:
                    continue
                if low in {k.lower() for k in self.forced}:
                    continue
                budget = 1 if len(low) <= 6 else 2
                hits: List[str] = []
                for length in range(len(low) - budget, len(low) + budget + 1):
                    for c_low, c_orig in by_len.get(length, []):
                        d = levenshtein_at_most(low, c_low, budget)
                        if 0 < d <= budget:
                            hits.append(c_orig)
                uniq = sorted(set(hits))
                if len(uniq) != 1:
                    continue
                seen.add(low)
                start = max(0, match.start() - 80)
                end = min(len(ch.text), match.end() + 80)
                candidates.append({
                    "token": token,
                    "canon": uniq[0],
                    "hits": freq.get(low, 0),
                    "context": re.sub(r"\s+", " ", ch.text[start:end]).strip(),
                })
        logger.info("Normalizer Tier 1: %d rare OOV candidates found", len(candidates))
        return candidates

    async def arbitrate(self, candidates: List[Dict[str, str]], llm_caller: Callable[..., Any], model_override: str) -> None:
        if not candidates:
            return
        prompt = (
            "You are a proofreader for an English novel.\n"
            "Each candidate is a rare token with one close canonical name.\n"
            "Return JSON only. Include a token SOLELY if it is an accidental typo of that name.\n"
            "If it is a nickname, diminutive (-y/-ie), wordplay, or a different word, omit it.\n"
            '{"typo_token": "CorrectedName"}'
        )
        try:
            result = await llm_caller(
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": json.dumps(candidates, ensure_ascii=False, indent=2)},
                ],
                temperature=0.1,
                max_tokens=1500,
                is_json=True,
                disable_thinking=True,
                stream=False,
                model_override=model_override,
            )
            data = json_repair.loads(Guardrails.extract_json_block(result.text))
            allowed = {c["token"]: c["canon"] for c in candidates}
            if isinstance(data, dict):
                for wrong, right in data.items():
                    canon = allowed.get(str(wrong).strip())
                    if canon and str(right).strip() == canon:
                        self.replacements[str(wrong).strip()] = canon
            logger.info("Normalizer Tier 2 (%s) approved %d fixes: %s",
                        model_override, len(self.replacements), self.replacements)
        except Exception as e:
            logger.warning("Normalizer arbitration skipped (%s)", e)

    def apply_to_all(self) -> None:
        merged = dict(self.replacements)
        merged.update(self.forced)
        if not merged:
            return
        alts = "|".join(re.escape(w) for w in sorted(merged, key=len, reverse=True))
        pattern = re.compile(rf"\b(?:{alts})('s?)?\b", re.IGNORECASE)

        def repl(match: re.Match) -> str:
            body = match.group(0)
            suffix = match.group(1) or ""
            key = strip_possessive(body)
            right = merged.get(key) or merged.get(key.lower())
            if right is None:
                for src, dst in merged.items():
                    if src.lower() == key.lower():
                        right = dst
                        break
            if not right:
                return body
            cased = right.upper() if key.isupper() else (right[:1].upper() + right[1:])
            return f"{cased}{suffix}"

        for ch in self.chapters:
            ch.text = pattern.sub(repl, ch.text)


# ==============================================================================
# 8. ДВИЖОК ЛОРА: ДИНАМИЧЕСКИЙ NER (~6K ТОКЕНОВ) И КУРАЦИЯ
# ==============================================================================

class LoreClusteringEngine:
    STOP_NOISE_REGEX = re.compile(
        r"(?i)("
        r"https?://\S+|www\.\S+|"
        r"\b(?:ao3|archiveofourown|fichub|fanfiction\.net|kofi|ko-fi|patreon|paypal|discord)\b|"
        r"\bchapter\s+\d+\b|"
        r"\bauthor'?s?\s+notes?\b|"
        r"\bdisclaimer\b|"
        r"^\s*[\d\W_]+\s*$"
        r")",
    )

    @classmethod
    def filter_noise(cls, raw_candidates: List[str], lexicon: Set[str]) -> List[str]:
        cleaned = []
        for c in raw_candidates:
            s = c.strip()
            if len(s) <= 1 or cls.STOP_NOISE_REGEX.search(s):
                continue
            # Отсекаем одиночные строчные слова из обычного словаря ("red", "death", "summer"),
            # чтобы они никогда не засоряли глоссарий как нарицательные слова.
            if " " not in s and "-" not in s and s.islower() and s in lexicon:
                continue
            cleaned.append(s)
        return cleaned

    @classmethod
    def cluster_similar_entities(cls, raw_entities: List[str], lexicon: Set[str], similarity_threshold: float = 0.82) -> List[List[str]]:
        sorted_terms = sorted(set(cls.filter_noise(raw_entities, lexicon)), key=lambda s: s.lower())
        clusters: List[List[str]] = []
        visited = set()
        for i, term in enumerate(sorted_terms):
            if term in visited:
                continue
            current = [term]
            visited.add(term)
            t_low = term.lower()
            t_pref = t_low[:3]
            for j in range(i + 1, len(sorted_terms)):
                cand = sorted_terms[j]
                if cand in visited:
                    continue
                c_low = cand.lower()
                is_sub = (t_low in c_low or c_low in t_low) and len(term) >= 4
                c_pref = c_low[:3]
                if not is_sub and c_pref and t_pref and c_pref[0] != t_pref[0] and not c_low.startswith(t_pref):
                    break
                if not is_sub and SequenceMatcher(None, t_low, c_low).ratio() < similarity_threshold:
                    continue
                current.append(cand)
                visited.add(cand)
            clusters.append(current)
        return clusters


class LoreAuditEngine:
    def __init__(self, llm: DualProtocolLLMClient, controller: AdaptiveController, cache_dir: Path):
        self.llm = llm
        self.controller = controller
        self.lexicon = EnglishLexicon.get_lexicon(cache_dir)

    async def profile_universe(self, book_title: str, chapters: List[ChapterSource]) -> UniverseProfile:
        logger.info("[0/3] Profiling universe via %s", self.llm.endpoint_cfg.model)
        sample = f"--- OPENING ---\n{chapters[0].text[:1500]}\n\n"
        if len(chapters) > 2:
            mid = len(chapters) // 2
            sample += f"--- MIDPOINT ---\n{chapters[mid].text[:1000]}"
        system_prompt = (
            "You are a localization director preparing a Russian edition of a long-form narrative.\n"
            "Infer the work's profile from the title and samples. Do not assume a franchise.\n"
            "Return JSON only:\n"
            "{\n"
            '  "work_type": "Original Work" | "Fandom Fanfiction" | "Published Translation",\n'
            '  "fandom": "Primary setting or series name, or Unknown",\n'
            '  "setting_details": "Genre, register, and world texture in one or two sentences",\n'
            '  "main_characters": {"Original Name": "F|M|N and short role"},\n'
            '  "key_lore_rules": ["Onomastic or translation constraints implied by the text"]\n'
            "}"
        )
        try:
            result = await self.llm.call(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Title: {book_title}\n\nSample:\n{sample}"},
                ],
                temperature=0.1,
                max_tokens=900,
                is_json=True,
                disable_thinking=False,
                stream=False,
            )
            data = json_repair.loads(Guardrails.extract_json_block(result.text))
            profile = UniverseProfile(
                work_type=data.get("work_type", "Original Work / Fanfiction"),
                fandom=data.get("fandom", "Unknown"),
                setting_details=data.get("setting_details", ""),
                main_characters=data.get("main_characters", {}),
                key_lore_rules=data.get("key_lore_rules", []),
            )
            logger.info("Profile: %s (%s)", profile.fandom, profile.work_type)
            return profile
        except Exception as e:
            logger.warning("Profiling failed (%s); using defaults", e)
            return UniverseProfile()

    async def harvest_full_book_with_ai(
        self,
        chapters: List[ChapterSource],
        max_workers: int = 2,
        model_override: str = "deepseek-v4.1-flash",
    ) -> List[str]:
        """Динамический сборщик сущностей крупными чанками (~4500 слов / ~6000 токенов) с авто-бисекцией."""
        logger.info("[1/3] Dynamic NER Harvesting via %s (~6k token super-batches)", model_override)
        batches: List[str] = []
        buf: List[str] = []
        buf_words = 0
        for ch in chapters:
            w = len(ch.text.split())
            if buf and buf_words + w > 4500:
                batches.append("\n\n=== NEXT CHAPTER ===\n\n".join(buf))
                buf, buf_words = [ch.text], w
            else:
                buf.append(ch.text)
                buf_words += w
        if buf:
            batches.append("\n\n=== NEXT CHAPTER ===\n\n".join(buf))

        total = len(batches)
        system_prompt = (
            "You are a Named Entity Recognition (NER) extractor for novel localization.\n"
            "Extract all universe-specific proper nouns, character names, capitalized nicknames, "
            "locations, organizations, named weapons, and coined lore terms.\n\n"
            "Rules:\n"
            "1. Extract full names AND informal capitalized nicknames (e.g. 'Ice Queen', 'Little Red', 'Torchwick').\n"
            "2. Do NOT extract lowercase common nouns, colors ('red'), seasons, days of the week, or generic words.\n"
            "3. Return JSON strictly matching this structure:\n"
            "{\n"
            '  "characters_and_nicknames": ["Full Name", "Capitalized Nickname"],\n'
            '  "locations_and_factions": ["Place", "Organization"],\n'
            '  "items_and_lore_terms": ["Weapon", "Species", "Coined Term"]\n'
            "}"
        )

        semaphore = asyncio.Semaphore(max_workers)
        harvested: Set[str] = set()
        done = 0
        lock = asyncio.Lock()

        async def scan_chunk(text: str, depth: int = 0) -> List[str]:
            try:
                result = await self.llm.call(
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": f"Extract entities from this narrative block:\n\n{text}"},
                    ],
                    temperature=0.1,
                    max_tokens=2000,
                    is_json=True,
                    disable_thinking=True,
                    stream=False,
                    model_override=model_override,
                )
                data = json_repair.loads(Guardrails.extract_json_block(result.text))
                items: List[str] = []
                if isinstance(data, dict):
                    for v in data.values():
                        if isinstance(v, list):
                            items.extend(str(x).strip() for x in v if isinstance(x, str) and len(x.strip()) > 1)
                elif isinstance(data, list):
                    items.extend(str(x).strip() for x in data if isinstance(x, str) and len(x.strip()) > 1)

                # Если чанк перенасыщен объектами (>= 45), делим пополам для предотвращения потери хвоста
                if len(items) >= HARVEST_SATURATION_LIMIT and depth < 2:
                    paras = text.split("\n\n")
                    if len(paras) >= 4:
                        logger.info("[harvest] Super-batch saturated (%d items); bisecting...", len(items))
                        mid = len(paras) // 2
                        h1 = await scan_chunk("\n\n".join(paras[:mid]), depth + 1)
                        h2 = await scan_chunk("\n\n".join(paras[mid:]), depth + 1)
                        return list(set(items + h1 + h2))
                return items
            except TruncationError:
                if depth < 2:
                    paras = text.split("\n\n")
                    if len(paras) >= 2:
                        mid = len(paras) // 2
                        h1 = await scan_chunk("\n\n".join(paras[:mid]), depth + 1)
                        h2 = await scan_chunk("\n\n".join(paras[mid:]), depth + 1)
                        return list(set(h1 + h2))
                return []
            except Exception as e:
                logger.warning("[harvest] Chunk scan failed: %s", e)
                return []

        async def worker(idx: int, text: str) -> None:
            nonlocal done
            async with semaphore:
                found = await scan_chunk(text)
                async with lock:
                    for item in found:
                        harvested.add(item)
                    done += 1
                    if done % 5 == 0 or done == total:
                        logger.info("[harvest] %d/%d super-batches complete (%d unique terms)", done, total, len(harvested))

        await asyncio.gather(*[asyncio.create_task(worker(i + 1, b)) for i, b in enumerate(batches)])
        return sorted(harvested)

    async def curate_master_lore(
        self,
        profile: UniverseProfile,
        candidates: List[str],
        pre_seeded: Dict[str, str],
    ) -> Dict[str, LoreEntry]:
        logger.info("[2/3] Curating glossary via %s", self.llm.endpoint_cfg.model)
        master: Dict[str, LoreEntry] = {}
        for k, v in pre_seeded.items():
            master[k.strip()] = LoreEntry(
                canonical_ru=v.strip(),
                surface_ru=v.strip(),
                gender="None",
                animacy="animate",
                category="custom",
                notes="pre-seeded surface",
            )
        if not candidates:
            return master
        flat = [item for cluster in LoreClusteringEngine.cluster_similar_entities(candidates, self.lexicon) for item in cluster]
        total = len(flat)
        system_prompt = (
            "You are a localization lexicographer preparing a Russian glossary.\n"
            f"Work: {profile.fandom} ({profile.work_type}). {profile.setting_details}\n"
            f"Prior rules: {json.dumps(profile.key_lore_rules, ensure_ascii=False)}\n\n"
            "Map each source string to its own Russian surface. Do not collapse nicknames into the full name.\n"
            "Rules:\n"
            "1. If a Russian fandom convention exists, use it.\n"
            "2. surface_ru is the form to emit when the source uses this exact string.\n"
            "3. canonical_ru is the lemma. For a nickname, canonical_ru is the person's lemma and surface_ru is the nickname.\n"
            "4. gender is the grammatical gender of surface_ru: F, M, N, or None.\n"
            "5. animacy is animate or inanimate.\n"
            "6. category is character, location, organization, item, race, idiom, or concept.\n"
            "7. notes must say when a string is a nickname or alias, or if it collides with a common English word.\n\n"
            "Return JSON:\n"
            '{ "Source": {"canonical_ru": "", "surface_ru": "", "gender": "F|M|N|None", '
            '"animacy": "animate|inanimate", "category": "character", "notes": ""} }'
        )
        cursor = 0
        batch_idx = 1
        fail_streak = 0
        while cursor < total:
            chunk = flat[cursor : cursor + self.controller.lore_batch_size]
            try:
                result = await self.llm.call(
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": "Candidates:\n" + json.dumps(chunk, ensure_ascii=False)},
                    ],
                    temperature=0.1,
                    max_tokens=1800,
                    is_json=True,
                    disable_thinking=False,
                    stream=False,
                )
                await self.controller.report_latency(result.elapsed, stage="lore")
                data = json_repair.loads(Guardrails.extract_json_block(result.text))
                if isinstance(data, dict):
                    for term, info in data.items():
                        if isinstance(info, dict):
                            surface = (info.get("surface_ru") or info.get("canonical_ru") or term).strip()
                            master[term.strip()] = LoreEntry(
                                canonical_ru=(info.get("canonical_ru") or surface).strip(),
                                surface_ru=surface,
                                gender=info.get("gender", "None"),
                                animacy=info.get("animacy", "inanimate"),
                                category=info.get("category", "concept"),
                                notes=info.get("notes", ""),
                            )
                logger.info("[lore] Batch %d ok in %.1fs (%d/%d)", batch_idx, result.elapsed, min(total, cursor + len(chunk)), total)
                cursor += len(chunk)
                batch_idx += 1
                fail_streak = 0
            except TransientNet as e:
                fail_streak += 1
                await self.controller.report_network_drop(stage="lore")
                logger.warning("[lore] Batch %d network fault (%s) %d/%d", batch_idx, str(e)[:80], fail_streak, MAX_LORE_BATCH_ATTEMPTS)
                if fail_streak >= MAX_LORE_BATCH_ATTEMPTS:
                    cursor += max(1, len(chunk))
                    batch_idx += 1
                    fail_streak = 0
                await asyncio.sleep(2.0)
            except Exception as e:
                fail_streak += 1
                logger.warning("[lore] Batch %d error (%s) %d/%d", batch_idx, str(e)[:80], fail_streak, MAX_LORE_BATCH_ATTEMPTS)
                if fail_streak >= MAX_LORE_BATCH_ATTEMPTS:
                    cursor += max(1, len(chunk))
                    batch_idx += 1
                    fail_streak = 0
                await asyncio.sleep(2.0)
            await asyncio.sleep(0.2)
        logger.info("Master lore ready: %d entries", len(master))
        return master


# ==============================================================================
# 9. ДВИЖОК ЛИТЕРАТУРНОГО ПЕРЕВОДА (С МАСКИРОВАНИЕМ КОЛЛИЗИЙ ГЛОССАРИЯ)
# ==============================================================================

class FastTranslationEngine:
    def __init__(self, llm: DualProtocolLLMClient, controller: AdaptiveController, cache_dir: Path, max_tokens: int = 6144):
        self.llm = llm
        self.controller = controller
        self.lexicon = EnglishLexicon.get_lexicon(cache_dir)
        self.max_tokens = max_tokens

    def filter_chapter_glossary(self, text: str, master_glossary: Dict[str, LoreEntry]) -> Dict[str, LoreEntry]:
        """
        Умный фильтр глоссария (Longest-Match Masking):
        1. Сначала ищет составные термины ("White Fang", "One Schnee Plaza") и маскирует их в буфере поиска,
           чтобы вложенные слова ("Fang", "Schnee Plaza") не срабатывали ложно.
        2. Одиночные слова, совпадающие со словарем английского языка ("Red", "Rose", "Summer", "Death"),
           матчуются СТРОГО с заглавной буквы, игнорируя нарицательные омонимы в нижнем регистре.
        """
        filtered: Dict[str, LoreEntry] = {}
        search_buf = text

        # Сортируем ключи от самых длинных к коротким
        sorted_keys = sorted(master_glossary.keys(), key=lambda k: len(k.strip()), reverse=True)

        for original in sorted_keys:
            src = original.strip()
            if not src:
                continue
            entry = master_glossary[original]

            # Пропускаем мусорные строчные словарные ключи, если они случайно попали из старого кэша
            if " " not in src and "-" not in src and src.islower() and src in self.lexicon:
                continue

            if (" " in src) or ("-" in src) or ("'" in src):
                pattern = re.compile(r"(?i)(?:\b|\A)" + re.escape(src) + r"(?:\b|\Z)")
                if pattern.search(search_buf):
                    filtered[original] = entry
                    # Маскируем найденную фразу пробелами, чтобы "White Fang" не триггерил одиночный "Fang"
                    search_buf = pattern.sub(" ", search_buf)
            else:
                # Если одиночное слово есть в обычном английском словаре (Red, Rose, Summer, Death, Fate),
                # требуем строгого совпадения регистра (с заглавной буквы)
                if src.lower() in self.lexicon or len(src) <= 4:
                    cap_src = src[:1].upper() + src[1:]
                    pattern = re.compile(r"\b" + re.escape(cap_src) + r"\b")
                else:
                    pattern = re.compile(r"(?i)\b" + re.escape(src) + r"\b")

                if pattern.search(search_buf):
                    filtered[original] = entry

        return filtered

    @staticmethod
    def _glossary_payload(chapter_glossary: Dict[str, LoreEntry]) -> Dict[str, Dict[str, str]]:
        payload = {}
        for original, entry in chapter_glossary.items():
            payload[original] = {
                "surface_ru": entry.use_form(),
                "lemma_ru": entry.canonical_ru,
                "gender": entry.gender,
                "animacy": entry.animacy,
                "category": entry.category,
                "notes": entry.notes,
            }
        return payload

    @staticmethod
    def _feminine_consonant_names(chapter_glossary: Dict[str, LoreEntry], segment: str) -> List[str]:
        names = []
        low = segment.lower()
        for original, entry in chapter_glossary.items():
            if entry.gender != "F":
                continue
            if original.lower() not in low and entry.use_form().lower() not in low:
                continue
            ru = entry.use_form()
            if not ru:
                continue
            if ru[-1].lower() not in RU_VOWELS and re.search(r"[а-яё]", ru, re.IGNORECASE):
                names.append(ru)
        return names

    def _state_card(self, segment: str, prev_source: str, chapter_glossary: Dict[str, LoreEntry]) -> str:
        seg_hits = self.filter_chapter_glossary(segment, chapter_glossary)
        present = [
            f"{orig} -> {entry.use_form()} (gender {entry.gender}, {entry.category})"
            for orig, entry in seg_hits.items()
        ]
        lines = []
        if present:
            lines.append("Active glossary in this segment: " + "; ".join(present[:35]))
        tail = (prev_source or "").rstrip()
        if tail.endswith(('"', "»", "—")) or "\n-" in tail[-120:]:
            lines.append("Previous source ended in ongoing dialogue. Continue it seamlessly.")
        return "\n".join(lines)

    def split_large_text(self, text: str, max_words: int) -> List[str]:
        if len(text.split()) <= max_words:
            return [text]
        paragraphs = text.split("\n\n")
        parts: List[str] = []
        current: List[str] = []
        current_words = 0
        for p in paragraphs:
            p_len = len(p.split())
            if p_len > max_words:
                if current:
                    parts.append("\n\n".join(current))
                    current, current_words = [], 0
                parts.extend(self._split_paragraph(p, max_words))
                continue
            if current_words + p_len > max_words and current:
                parts.append("\n\n".join(current))
                current = [p]
                current_words = p_len
            else:
                current.append(p)
                current_words += p_len
        if current:
            parts.append("\n\n".join(current))
        return parts

    @staticmethod
    def _split_paragraph(paragraph: str, max_words: int) -> List[str]:
        sentences = _SENT_SPLIT.split(paragraph)
        parts: List[str] = []
        buf: List[str] = []
        n = 0
        for s in sentences:
            w = len(s.split())
            if n + w > max_words and buf:
                parts.append(" ".join(buf))
                buf, n = [s], w
            else:
                buf.append(s)
                n += w
        if buf:
            parts.append(" ".join(buf))
        return parts

    @staticmethod
    def _russian_tail(translated: str, max_paras: int = 2) -> str:
        paras = [p for p in translated.split("\n\n") if p.strip()]
        return "\n\n".join(paras[-max_paras:]) if paras else translated[-700:]

    def _system_prompt(self, profile: UniverseProfile, feminine: List[str]) -> str:
        feminine_hint = ""
        if feminine:
            feminine_hint = (
                " Feminine past-tense verbs are mandatory for these consonant-ending female names: "
                + ", ".join(feminine) + "."
            )
        return (
            "You are an acclaimed literary translator producing an authentic Russian edition of narrative prose.\n"
            f"Setting: {profile.fandom}. Register: {profile.setting_details}\n\n"
            "Produce Russian that reads as genuine original prose, avoiding mechanical translation calques.\n"
            "Rules:\n"
            "1. Use the glossary's surface_ru when a word is used as that specific proper noun or nickname. "
            "However, if a word (like 'red', 'rose', 'summer', 'death') is used in the sentence as an ordinary "
            "common noun, verb, or adjective (e.g. 'flashed red', 'rose to his feet'), translate it naturally by meaning!\n"
            "2. Preserve nicknames as nicknames; do not replace them with full formal names.\n"
            "3. Match the author's tone. Recast English syntax: drop needless possessives (свой/его/её), prefer active finite verbs.\n"
            "4. Adapt wordplay and puns naturally. Keep intentional latinized incantations (e.g. 'Vliyaniye') in italics.\n"
            "5. Avoid anachronistic LitRPG/gamer jargon in urban procedural fantasy.\n"
            f"6. Gender agreement follows the glossary.{feminine_hint}\n"
            "7. Dialogue lines MUST start with an em dash and a space (— ). Preserve Markdown (*italics*, **bold**, * * *).\n"
            "8. Do not bowdlerize adult content, profanity, or violence.\n"
            "9. If a source overlap is provided, do not translate it again.\n"
            "Output only the completed Russian translation."
        )

    async def _translate_segment_once(
        self,
        segment_text: str,
        profile: UniverseProfile,
        chapter_glossary: Dict[str, LoreEntry],
        previous_ru: str = "",
        previous_source: str = "",
    ) -> LLMResult:
        seg_glossary = self.filter_chapter_glossary(segment_text, chapter_glossary)
        feminine = self._feminine_consonant_names(seg_glossary, segment_text)
        overlap = ""
        if previous_source.strip():
            overlap = (
                "--- SOURCE OVERLAP (already translated; do not output this again) ---\n"
                f"{tail_sentences(previous_source, 2)}\n"
                "--- END OVERLAP ---\n\n"
            )
        ru_ctx = ""
        if previous_ru.strip():
            ru_ctx = (
                "--- PRECEDING RUSSIAN (match gender and tempo; do not retranslate) ---\n"
                f"{previous_ru}\n"
                "--- END RUSSIAN ---\n\n"
            )
        state = self._state_card(segment_text, previous_source, seg_glossary)
        state_block = f"State card:\n{state}\n\n" if state else ""
        user_content = (
            "Segment glossary:\n"
            f"{json.dumps(self._glossary_payload(seg_glossary), ensure_ascii=False)}\n\n"
            f"{state_block}{ru_ctx}{overlap}"
            f"SOURCE:\n{segment_text}\n\n"
            "Russian translation:"
        )
        result = await self.llm.call(
            messages=[
                {"role": "system", "content": self._system_prompt(profile, feminine)},
                {"role": "user", "content": user_content},
            ],
            temperature=0.3,
            max_tokens=self.max_tokens,
            is_json=False,
            disable_thinking=True,
            stream=True,
        )
        await self.controller.report_latency(result.elapsed, stage="trans")
        return result

    async def _translate_segment(
        self,
        segment_text: str,
        profile: UniverseProfile,
        chapter_glossary: Dict[str, LoreEntry],
        previous_ru: str = "",
        previous_source: str = "",
        depth: int = 0,
    ) -> LLMResult:
        try:
            return await self._translate_segment_once(
                segment_text, profile, chapter_glossary, previous_ru, previous_source,
            )
        except TruncationError:
            if depth >= 3:
                raise RuntimeError("Segment still truncates after 3 splits")
            halves = self.split_large_text(segment_text, max(120, len(segment_text.split()) // 2))
            if len(halves) == 1:
                halves = self._split_paragraph(segment_text, max(80, len(segment_text.split()) // 2))
            if len(halves) == 1:
                raise
            logger.warning("Token cap: splitting %d words into %d", len(segment_text.split()), len(halves))
            pieces: List[str] = []
            tail = previous_ru
            src_tail = previous_source
            model = self.llm.endpoint_cfg.model
            for half in halves:
                piece = await self._translate_segment(half, profile, chapter_glossary, tail, src_tail, depth + 1)
                pieces.append(piece.text)
                tail = self._russian_tail(piece.text)
                src_tail = half
                model = piece.model
            return LLMResult("\n\n".join(pieces), 0.0, model, "")

    async def translate_chapter(
        self,
        idx: int,
        raw_text: str,
        profile: UniverseProfile,
        chapter_glossary: Dict[str, LoreEntry],
        cache: CacheManager,
        semaphore: asyncio.Semaphore,
        on_complete: Callable[[int], Any],
    ) -> Tuple[int, str]:
        source_sha = segment_fingerprint(raw_text)
        cached = cache.load_translation(idx, source_sha)
        if cached:
            await on_complete(idx)
            return idx, cached

        chunk_words = self.controller.snapshot_chunk()
        segments = self.split_large_text(raw_text, chunk_words)
        already = cache.load_partial_segments(idx, segments)
        model = self.llm.endpoint_cfg.model

        async with semaphore:
            translated = list(already)
            previous_ru = self._russian_tail(already[-1]) if already else ""
            previous_source = segments[len(already) - 1] if already else ""
            if already:
                logger.info("Chapter %d: resuming at segment %d/%d", idx, len(already) + 1, len(segments))

            for seg_i, seg in enumerate(segments):
                if seg_i < len(translated):
                    continue
                try:
                    res = await self._translate_segment(seg, profile, chapter_glossary, previous_ru, previous_source)
                except ContentFilterError as cfe:
                    logger.error("CENSORSHIP TRIGGER in Chapter %d: %s", idx, cfe)
                    raise ChapterTriggerError(
                        chapter_idx=idx,
                        error_type="CONTENT_FILTER",
                        model=model,
                        snippet=seg[:280],
                        raw_error=str(cfe),
                    ) from cfe
                except TransientNet as net_err:
                    await self.controller.report_network_drop(stage="trans")
                    if translated:
                        cache.save_partial_segments(idx, translated, segments, model)
                    raise net_err
                except Exception as e:
                    if translated:
                        cache.save_partial_segments(idx, translated, segments, model)
                    if Guardrails.is_refusal(str(e)):
                        raise ChapterTriggerError(
                            chapter_idx=idx,
                            error_type="REFUSAL",
                            model=model,
                            snippet=seg[:280],
                            raw_error=str(e),
                        ) from e
                    raise

                translated.append(res.text)
                previous_ru = self._russian_tail(res.text)
                previous_source = seg
                model = res.model
                cache.save_partial_segments(idx, translated, segments, model)

            full = "\n\n".join(translated)

        cache.save_translation(idx, full, model, source_sha)
        await on_complete(idx)
        return idx, full


# ==============================================================================
# 10. ГЛАВНЫЙ ОРКЕСТРАТОР ПАЙПЛАЙНА
# ==============================================================================

class MasterTranslationPipeline:
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.stats = StatsTracker()
        self.controller = AdaptiveController(
            max_workers=config.max_concurrent_translations,
        )
        self.curator_llm = DualProtocolLLMClient(
            endpoint_cfg=config.curator,
            timeout=config.request_timeout,
            max_retries=config.max_network_retries,
            backoff=config.backoff_factor,
            stats=self.stats,
        )
        self.translator_llm = DualProtocolLLMClient(
            endpoint_cfg=config.translator,
            timeout=config.request_timeout,
            max_retries=config.max_network_retries,
            backoff=config.backoff_factor,
            stats=self.stats,
        )
        self.lore_engine = LoreAuditEngine(self.curator_llm, self.controller, config.cache_dir)
        self.translator = FastTranslationEngine(
            self.translator_llm, self.controller, config.cache_dir, max_tokens=config.translation_max_tokens,
        )

    async def run(
        self,
        input_epub: Path,
        output_epub: Optional[Path] = None,
        max_workers: Optional[int] = None,
        pre_seeded_glossary: Optional[Dict[str, str]] = None,
        force_chapters: Optional[Set[int]] = None,
        progress_cb: Optional[Callable[[float, str], None]] = None,
    ) -> Tuple[Path, str]:
        try:
            return await self._run_inner(
                input_epub, output_epub, max_workers, pre_seeded_glossary, force_chapters, progress_cb,
            )
        finally:
            await self.curator_llm.aclose()
            await self.translator_llm.aclose()

    async def _translate_titles(self, chapters: List[ChapterSource], glossary: Dict[str, LoreEntry]) -> None:
        pending = [c.title for c in chapters if re.search(r"[A-Za-z]", c.title or "")]
        if not pending:
            return
        subset = {
            k: v.use_form()
            for k, v in glossary.items()
            if any(k.lower() in title.lower() for title in pending)
        }
        prompt = (
            "Translate these chapter titles into Russian. Use the glossary surfaces. "
            "Keep 'Пролог' if the title is a prologue. Return JSON object mapping original title to Russian title."
        )
        try:
            result = await self.curator_llm.call(
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": json.dumps({"titles": pending, "glossary": subset}, ensure_ascii=False)},
                ],
                temperature=0.1,
                max_tokens=1500,
                is_json=True,
                disable_thinking=False,
                stream=False,
            )
            data = json_repair.loads(Guardrails.extract_json_block(result.text))
            mapping = data.get("titles") if isinstance(data, dict) and isinstance(data.get("titles"), dict) else data
            if not isinstance(mapping, dict):
                return
            for ch in chapters:
                ru = mapping.get(ch.title)
                if isinstance(ru, str) and ru.strip():
                    ch.title = ru.strip()
        except Exception as e:
            logger.warning("Title batch skipped (%s)", e)

    async def _run_inner(
        self,
        input_epub: Path,
        output_epub: Optional[Path],
        max_workers: Optional[int],
        pre_seeded_glossary: Optional[Dict[str, str]],
        force_chapters: Optional[Set[int]],
        progress_cb: Optional[Callable[[float, str], None]],
    ) -> Tuple[Path, str]:
        if progress_cb:
            progress_cb(0.02, "Probing endpoint...")
        self.controller.calibrate(await self.translator_llm.run_reference_probe())

        book_title, raw_chapters = EpubService.extract_chapters_and_metadata(input_epub)
        total = len(raw_chapters)
        if total == 0:
            raise ValueError("No chapters extracted from the EPUB")

        workers_count = max_workers or self.config.max_concurrent_translations
        semaphore = asyncio.Semaphore(workers_count)
        path_hash = hashlib.sha1(str(input_epub.resolve()).encode("utf-8")).hexdigest()[:8]
        cache = CacheManager(
            self.config.cache_dir,
            f"{input_epub.stem}_{path_hash}",
            self.config.translator.model,
        )
        if force_chapters:
            cache.invalidate_chapters(force_chapters)

        # Проверяем, есть ли уже очищенные от аннотаций и опечаток главы в кэше
        cached_preprocessed = cache.load_preprocessed_chapters(total)
        cached_lore = cache.load_master_lore(
            self.config.curator.model,
            allow_legacy=self.config.reuse_existing_lore,
        )

        if cached_lore and cached_preprocessed:
            profile, master_glossary = cached_lore
            raw_chapters = cached_preprocessed
            logger.info("Loaded cached Master Lore (%d terms) and preprocessed chapters.", len(master_glossary))
            if progress_cb:
                progress_cb(0.19, f"Lore & Preprocessed text loaded ({profile.fandom})")
        else:
            # ЭТАП 0.5: ЧИСТКА АВТОРСКИХ АННОТАЦИЙ ЧЕРЕЗ ЯКОРЯ (DEEPSEEK-V4.1-FLASH)
            if progress_cb:
                progress_cb(0.04, f"Cleaning Author Notes ({self.config.harvester_model})...")
            await LLMAnnotationCleaner.clean_chapters(
                chapters=raw_chapters,
                llm_caller=self.curator_llm.call,
                model=self.config.harvester_model,
            )

            # ЭТАП 1..2: MASTER LORE
            if cached_lore:
                profile, master_glossary = cached_lore
                logger.info("Reusing existing Master Lore (%s, %d terms)", profile.fandom, len(master_glossary))
            else:
                if progress_cb:
                    progress_cb(0.07, "Profiling universe...")
                profile = await self.lore_engine.profile_universe(book_title, raw_chapters)

                if progress_cb:
                    progress_cb(0.10, f"Dynamic NER Harvesting ({self.config.harvester_model})...")
                candidates = await self.lore_engine.harvest_full_book_with_ai(
                    raw_chapters, max_workers=workers_count, model_override=self.config.harvester_model,
                )

                if progress_cb:
                    progress_cb(0.15, f"Curating glossary ({self.config.curator.model})...")
                master_glossary = await self.lore_engine.curate_master_lore(
                    profile, candidates, pre_seeded_glossary or {},
                )
                cache.save_master_lore(profile, master_glossary, curator_model=self.config.curator.model)

            # ЭТАП 2.5: НОРМАЛИЗАТОР ОПЕЧАТОК (СЛОВАРЬ 370K + GPT-6-LUNA)
            if progress_cb:
                progress_cb(0.18, f"Arbitrating rare typos ({self.config.arbiter_model})...")
            normalizer = SourceTypoNormalizer(
                master_glossary, raw_chapters, self.config.cache_dir, forced=pre_seeded_glossary or {},
            )
            oov_candidates = normalizer.harvest_oov_candidates()
            if oov_candidates:
                await normalizer.arbitrate(oov_candidates, self.curator_llm.call, model_override=self.config.arbiter_model)
            normalizer.apply_to_all()

            if progress_cb:
                progress_cb(0.19, "Translating chapter titles...")
            await self._translate_titles(raw_chapters, master_glossary)
            cache.save_preprocessed_chapters(raw_chapters)

        self.stats.total_words_original = sum(len(c.text.split()) for c in raw_chapters)

        # ЭТАП 3: ПОТОКОВЫЙ ХУДОЖЕСТВЕННЫЙ ПЕРЕВОД
        logger.info(
            "[3/3] Translating %d chapters x %d workers via %s (%s)",
            total, workers_count, self.config.translator.model, self.config.translator.resolved_protocol(),
        )
        self.stats.mark_translate_start()
        completed = 0
        done_lock = asyncio.Lock()

        async def mark_done(idx: int) -> None:
            nonlocal completed
            async with done_lock:
                completed += 1
                current = completed
            logger.info("[ok] Chapter %d (%d/%d)", idx, current, total)
            if progress_cb:
                progress_cb(0.20 + 0.80 * (current / total), f"Translated {current}/{total}")

        tasks = [
            asyncio.create_task(self.translator.translate_chapter(
                idx=i,
                raw_text=chapter.text,
                profile=profile,
                chapter_glossary=self.translator.filter_chapter_glossary(chapter.text, master_glossary),
                cache=cache,
                semaphore=semaphore,
                on_complete=mark_done,
            ))
            for i, chapter in enumerate(raw_chapters, 1)
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        translated_chapters: List[ChapterSource] = []
        failed_triggers: List[ChapterTriggerError] = []
        generic_failures: List[Tuple[int, Exception]] = []

        for i, res in enumerate(results, 1):
            if isinstance(res, ChapterTriggerError):
                failed_triggers.append(res)
            elif isinstance(res, Exception):
                generic_failures.append((i, res))
            else:
                _, text = res
                src = raw_chapters[i - 1]
                translated_chapters.append(ChapterSource(
                    title=src.title,
                    text=TextFormatter.post_lint(text),
                    spine_id=src.spine_id,
                ))

        if failed_triggers or generic_failures:
            report_lines = ["\n❌ ПАЙПЛАЙН ОСТАНОВЛЕН: ОБНАРУЖЕНЫ СБОИ ИЛИ ТРИГГЕРЫ ЦЕНЗУРЫ:\n"]
            for err in failed_triggers:
                report_lines.append(
                    f"🛑 [ГЛАВА {err.chapter_idx}] ТИП: {err.error_type} | МОДЕЛЬ: {err.model}\n"
                    f"   Ответ: {err.raw_error}\n"
                    f"   Триггер-текст (фрагмент):\n"
                    f"   \"{err.snippet}...\"\n"
                )
            for ch_idx, exc in generic_failures:
                report_lines.append(f"⚠️ [ГЛАВА {ch_idx}] Сбой: {exc}\n")

            full_report = "\n".join(report_lines)
            logger.error(full_report)
            raise RuntimeError(full_report)

        if output_epub is None:
            output_epub = input_epub.parent / f"{input_epub.stem}_RU.epub"
        EpubService.build_epub(translated_chapters, output_epub, f"{book_title} (RU)")
        summary = self.stats.get_summary()
        logger.info("Done: %s", summary)
        if progress_cb:
            progress_cb(1.0, f"Done. {summary}")
        return output_epub, summary


# ==============================================================================
# 11. WEB GUI (GRADIO) И CLI
# ==============================================================================

def parse_glossary_input(raw_glossary: str) -> Dict[str, str]:
    if not raw_glossary or not raw_glossary.strip():
        return {}
    raw_glossary = raw_glossary.strip()
    if raw_glossary.startswith("{"):
        try:
            return json.loads(raw_glossary)
        except Exception:
            pass
    glossary = {}
    for line in raw_glossary.splitlines():
        line = line.strip()
        if "=" in line:
            k, v = line.split("=", 1)
        elif ":" in line:
            k, v = line.split(":", 1)
        else:
            continue
        glossary[k.strip()] = v.strip()
    return glossary


def parse_chapter_ranges(raw_str: str) -> Set[int]:
    if not raw_str or not raw_str.strip():
        return set()
    result: Set[int] = set()
    for part in raw_str.split(","):
        part = part.strip()
        if "-" in part:
            try:
                start, end = map(int, part.split("-"))
                result.update(range(start, end + 1))
            except ValueError:
                pass
        else:
            try:
                result.add(int(part))
            except ValueError:
                pass
    return result


def launch_web_gui() -> None:
    import gradio as gr

    def clean_str(val: Any) -> str:
        return str(val or "").strip()

    async def check_models(url: Any, key: Any):
        u = clean_str(url)
        k = clean_str(key)
        if not u:
            return "Set a base URL"
        models = await DualProtocolLLMClient.fetch_available_models(u, k)
        if not models:
            return "Could not list models."
        return "Available models:\n" + "\n".join(f"• {m}" for m in models)

    async def web_process(
        epub_file,
        trans_url, trans_key, trans_m, trans_proto,
        curator_url, curator_key, curator_m, curator_proto,
        harvester_m, arbiter_m, reuse_lore,
        workers, custom_glossary_raw, force_chapters_raw,
        progress=gr.Progress(),
    ):
        if not epub_file:
            return None, "Upload an .epub file."

        t_url_clean = clean_str(trans_url) or "https://cerberus-api.ddns.net/v1"
        t_key_clean = clean_str(trans_key)
        c_url_clean = clean_str(curator_url) or t_url_clean
        c_key_clean = clean_str(curator_key) or t_key_clean

        cfg = PipelineConfig(
            max_concurrent_translations=int(workers or 2),
            harvester_model=clean_str(harvester_m) or "deepseek-v4.1-flash",
            arbiter_model=clean_str(arbiter_m) or "gpt-6-luna",
            reuse_existing_lore=bool(reuse_lore),
            curator=EndpointConfig(
                base_url=c_url_clean,
                api_key=c_key_clean,
                model=clean_str(curator_m) or "claude-sonnet-5",
                protocol=clean_str(curator_proto) or "auto",
            ),
            translator=EndpointConfig(
                base_url=t_url_clean,
                api_key=t_key_clean,
                model=clean_str(trans_m) or "claude-sonnet-5",
                protocol=clean_str(trans_proto) or "auto",
            ),
        )
        pipeline = MasterTranslationPipeline(config=cfg)

        def cb(frac, desc):
            progress(frac, desc=desc)

        try:
            out_path, stats = await pipeline.run(
                input_epub=Path(epub_file.name),
                max_workers=int(workers or 2),
                pre_seeded_glossary=parse_glossary_input(clean_str(custom_glossary_raw)),
                force_chapters=parse_chapter_ranges(clean_str(force_chapters_raw)),
                progress_cb=cb,
            )
            return str(out_path), f"Finished.\n\n{stats}"
        except Exception as e:
            return None, f"{e}"

    with gr.Blocks(title="Cerberus Enterprise Translator v10.0") as ui:
        gr.Markdown("## Cerberus Enterprise Literary Translator v10.0")
        gr.Markdown(
            "**Multi-Stage Routing:** Annotation Cleaner & Dynamic NER (`deepseek-v4.1-flash`, x0.3) $\\to$ "
            "Lexicon + Typo Arbiter (`gpt-6-luna`, x0.38) $\\to$ "
            "Lore & Literary Prose (`claude-sonnet-5`, x2.0)"
        )
        with gr.Row():
            with gr.Column():
                f_in = gr.File(label="Source EPUB", file_types=[".epub"])
                with gr.Row():
                    w_in = gr.Slider(minimum=1, maximum=4, value=2, step=1, label="Parallel workers", scale=2)
                    reuse_lore_chk = gr.Checkbox(label="Reuse existing master_lore.json (if present)", value=True, scale=1)

                with gr.Accordion("Core Translator & API Key", open=True):
                    t_url = gr.Textbox(label="Base URL", value="https://cerberus-api.ddns.net/v1")
                    t_key = gr.Textbox(label="API key", value="", type="password", placeholder="sk-...")
                    with gr.Row():
                        t_m = gr.Dropdown(label="Translator Model", choices=CERBERUS_MODELS, value="claude-sonnet-5", allow_custom_value=True, scale=3)
                        t_proto = gr.Dropdown(label="Protocol", choices=["auto", "anthropic", "openai"], value="auto", scale=1)
                    check_btn = gr.Button("List gateway models")
                    models_log = gr.Textbox(label="/models", value="", interactive=False, lines=3)

                with gr.Accordion("Cost-Efficiency Stage Routing (Lore / Harvester / Arbiter)", open=False):
                    with gr.Row():
                        harv_m = gr.Dropdown(label="Stage 0.5 & 1: Annotation & NER Harvester", choices=CERBERUS_MODELS, value="deepseek-v4.1-flash", allow_custom_value=True)
                        arb_m = gr.Dropdown(label="Stage 2.5: Typo Arbiter", choices=CERBERUS_MODELS, value="gpt-6-luna", allow_custom_value=True)
                    c_url = gr.Textbox(label="Curator Base URL (leave default to match Translator)", value="https://cerberus-api.ddns.net/v1")
                    c_key = gr.Textbox(label="Curator API key (optional if same as Translator)", value="", type="password")
                    with gr.Row():
                        c_m = gr.Dropdown(label="Stage 2: Lore Curator Model", choices=CERBERUS_MODELS, value="claude-sonnet-5", allow_custom_value=True)
                        c_proto = gr.Dropdown(label="Protocol", choices=["auto", "anthropic", "openai"], value="auto")

                g_in = gr.Textbox(label="Pre-seeded glossary", value="", lines=3, placeholder="Original Name = Русское имя")
                ch_in = gr.Textbox(label="Redo chapters", value="", placeholder="44-45, 48-54")
                btn = gr.Button("Start Translation Pipeline", variant="primary")

            with gr.Column():
                st = gr.Textbox(label="Status / Diagnostic Output", value="", interactive=False, lines=14)
                f_out = gr.File(label="Translated EPUB")

        check_btn.click(check_models, inputs=[t_url, t_key], outputs=[models_log])
        btn.click(
            web_process,
            inputs=[
                f_in, t_url, t_key, t_m, t_proto,
                c_url, c_key, c_m, c_proto,
                harv_m, arb_m, reuse_lore_chk,
                w_in, g_in, ch_in,
            ],
            outputs=[f_out, st],
        )
    ui.launch(inbrowser=True)

def main() -> None:
    parser = argparse.ArgumentParser(description="Cerberus Enterprise Translator v10.0")
    parser.add_argument("--file", type=Path)
    parser.add_argument("--trans-url", default="https://cerberus-api.ddns.net/v1")
    parser.add_argument("--trans-key", default="")
    parser.add_argument("--trans-model", default="claude-sonnet-5")
    parser.add_argument("--trans-protocol", default="auto")
    parser.add_argument("--curator-url", default="https://cerberus-api.ddns.net/v1")
    parser.add_argument("--curator-key", default="")
    parser.add_argument("--curator-model", default="claude-sonnet-5")
    parser.add_argument("--curator-protocol", default="auto")
    parser.add_argument("--harvester-model", default="deepseek-v4.1-flash")
    parser.add_argument("--arbiter-model", default="gpt-6-luna")
    parser.add_argument("--rebuild-lore", action="store_true")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--glossary", default="")
    parser.add_argument("--chapters", default="")
    parser.add_argument("--list-models", action="store_true")
    parser.add_argument("--gui", action="store_true")
    args = parser.parse_args()

    if args.list_models:
        async def list_m():
            models = await DualProtocolLLMClient.fetch_available_models(args.trans_url, args.trans_key)
            if models:
                print("Models:")
                for m in models:
                    print(f"  • {m}")
            else:
                print("No models returned.")
        asyncio.run(list_m())
        return

    if args.gui or not args.file:
        launch_web_gui()
        return

    if args.glossary:
        g_path = Path(args.glossary)
        pre_glossary = json.loads(g_path.read_text(encoding="utf-8")) if g_path.exists() else parse_glossary_input(args.glossary)
    else:
        pre_glossary = {}

    cfg = PipelineConfig(
        max_concurrent_translations=args.workers,
        harvester_model=args.harvester_model,
        arbiter_model=args.arbiter_model,
        reuse_existing_lore=not args.rebuild_lore,
        curator=EndpointConfig(
            base_url=args.curator_url,
            api_key=args.curator_key or args.trans_key,
            model=args.curator_model,
            protocol=args.curator_protocol,
        ),
        translator=EndpointConfig(
            base_url=args.trans_url,
            api_key=args.trans_key,
            model=args.trans_model,
            protocol=args.trans_protocol,
        ),
    )
    pipeline = MasterTranslationPipeline(config=cfg)

    def cli_cb(frac, desc):
        print(f"[{int(frac * 100):02d}%] {desc}")

    try:
        out, stats = asyncio.run(pipeline.run(
            input_epub=args.file,
            max_workers=args.workers,
            pre_seeded_glossary=pre_glossary,
            force_chapters=parse_chapter_ranges(args.chapters),
            progress_cb=cli_cb,
        ))
        print(f"\n[ok] {out}\n{stats}")
    except KeyboardInterrupt:
        print("\nStopped. Cache kept.")
        raise SystemExit(130)


if __name__ == "__main__":
    main()