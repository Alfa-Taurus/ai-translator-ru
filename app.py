#!/usr/bin/env python3
"""
AI Literary Novel Translator (Async Streaming Pipeline v3.7 - Stable Engine)
- Полностью удален нестабильный V4 Flash
- Основная модель: DeepSeek V3.2 (без цензуры, отличный русский слог)
- Защита от сжигания входных токенов (Fast Failover при сбоях)
- Полная динамическая грамматика рода и сохранение форматирования
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import random
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import ebooklib
from bs4 import BeautifulSoup
from ebooklib import epub
import json_repair
from openai import AsyncOpenAI

# ==========================================
# 0. ЛОГИРОВАНИЕ
# ==========================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("TranslatorPipeline")


# ==========================================
# 1. КОНФИГУРАЦИЯ И СТАТИСТИКА
# ==========================================

@dataclass(frozen=True)
class PipelineConfig:
    """Конфигурация пайплайна."""
    base_url: str = "https://openrouter.ai/api/v1"
    cache_dir: Path = Path("./translation_cache")
    request_timeout: float = 90.0
    max_retries_per_model: int = 2
    backoff_factor: float = 1.5
    max_concurrent_translations: int = 15

    # Фаза 1: Анализ сюжета, извлечение терминов и пола персонажей
    models_pass1: List[str] = field(default_factory=lambda: [
        "deepseek/deepseek-v3.2",
    ])

    # Фаза 2: Художественный перевод
    models_pass2: List[str] = field(default_factory=lambda: [
        "deepseek/deepseek-v3.2",
    ])


@dataclass
class StatsTracker:
    """Сбор метрик расхода токенов, скорости и времени."""
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    total_words_original: int = 0
    start_time: float = field(default_factory=time.time)
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    async def add_usage(self, prompt_tokens: int, completion_tokens: int) -> None:
        async with self._lock:
            self.total_prompt_tokens += prompt_tokens
            self.total_completion_tokens += completion_tokens

    def get_summary(self) -> str:
        elapsed = max(1.0, time.time() - self.start_time)
        mins, secs = int(elapsed // 60), int(elapsed % 60)
        total_tokens = self.total_prompt_tokens + self.total_completion_tokens
        wpm = int((self.total_words_original / elapsed) * 60) if self.total_words_original else 0
        return (
            f"⏱ Время: {mins}м {secs}с | "
            f"📊 Токены: {total_tokens:,} (Вход: {self.total_prompt_tokens:,}, Выход: {self.total_completion_tokens:,}) | "
            f"🚀 Скорость: ~{wpm:,} слов/мин"
        )


@dataclass
class ChapterMeta:
    """Контекст и накопленный глоссарий для главы."""
    summary: str
    glossary: Dict[str, str] = field(default_factory=dict)


# ==========================================
# 2. ФОРМАТИРОВАНИЕ И БЕЗОПАСНЫЙ ВАЛИДАТОР
# ==========================================

class TextFormatter:
    """Конвертер HTML <-> Markdown для сохранения авторского курсива и оформления."""

    @staticmethod
    def html_to_markdown(html_content: str) -> str:
        soup = BeautifulSoup(html_content, "html.parser")

        for em in soup.find_all(["i", "em"]):
            em.replace_with(f"*{em.get_text()}*")
        for bold in soup.find_all(["b", "strong"]):
            bold.replace_with(f"**{bold.get_text()}**")
        for hr in soup.find_all("hr"):
            hr.replace_with("\n\n* * *\n\n")

        paragraphs = []
        for elem in soup.find_all(["p", "h1", "h2", "h3", "h4", "div", "blockquote"]):
            text = elem.get_text().strip()
            if text:
                paragraphs.append(text)

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


class Guardrails:
    """Безопасный валидатор без ложных срабатываний на русский язык."""

    _REFUSAL_REGEX = re.compile(
        r"(i cannot fulfill|i am unable to|as an ai language model|"
        r"my safety policies|i must decline|violates safety guidelines|content policy|"
        r"无法给到相关内容|不符合安全规范)",
        re.IGNORECASE,
    )

    # Детектор реальных мутаций сэмплирования (-аллка, -овалка)
    _MUTANT_VERBS_REGEX = re.compile(r"\b[а-яА-ЯёЁ]{3,}(?:аллка|овалка|стоналлка|ложилка|получилка|деформовалка)\b", re.IGNORECASE)

    # Детектор словарных петель (30+ слов подряд без знаков препинания)
    _ZERO_PUNCTUATION_LOOP_REGEX = re.compile(r"(?:\b[а-яА-ЯёЁa-zA-Z0-9-]+\b\s+){30,}[а-яА-ЯёЁa-zA-Z0-9-]+")

    # Детектор непереведенных кусков на английском (15+ английских слов подряд)
    _ENGLISH_LEAK_REGEX = re.compile(r"(?:\b[a-zA-Z]+\b[\s,;]+){15,}")

    @classmethod
    def is_refusal(cls, text: Optional[str]) -> bool:
        if not text:
            return True
        return bool(cls._REFUSAL_REGEX.search(text))

    @classmethod
    def is_degenerated(cls, text: str) -> bool:
        if not text or len(text) < 60:
            return False

        if cls._MUTANT_VERBS_REGEX.search(text):
            logger.warning("[Guardrail] Зафиксированы мутировавшие суффиксы (-лка).")
            return True

        if cls._ZERO_PUNCTUATION_LOOP_REGEX.search(text):
            logger.warning("[Guardrail] Зафиксирована словарная петля (Dictionary Loop).")
            return True

        if cls._ENGLISH_LEAK_REGEX.search(text):
            logger.warning("[Guardrail] Зафиксирован непереведенный блок на английском.")
            return True

        if re.search(r"(.{30,})\1{4,}", text):
            logger.warning("[Guardrail] Зафиксировано бесконечное зацикливание текста.")
            return True

        return False

    @classmethod
    def is_valid_translation(cls, text: Optional[str]) -> bool:
        if not text or len(text.strip()) < 10:
            return False
        if cls.is_refusal(text):
            return False
        if cls.is_degenerated(text):
            return False
        return True

    @staticmethod
    def extract_json_block(raw_text: str) -> str:
        if not raw_text:
            return ""
        start = raw_text.find("{")
        end = raw_text.rfind("}")
        if start != -1 and end != -1 and end > start:
            return raw_text[start : end + 1]
        return raw_text

    @classmethod
    def normalize_and_merge_glossary(
        cls, existing: Dict[str, str], new_terms: Dict[str, str]
    ) -> Dict[str, str]:
        key_map = {k.strip().lower(): k for k in existing}
        merged = dict(existing)
        for raw_k, raw_v in new_terms.items():
            if isinstance(raw_k, str) and isinstance(raw_v, str):
                k = raw_k.strip()
                v = raw_v.strip()
                if len(k) > 1 and len(v) > 1:
                    k_lower = k.lower()
                    if k_lower not in key_map:
                        merged[k] = v
                        key_map[k_lower] = k
        return merged

    @classmethod
    def parse_analysis(
        cls, raw_text: str, fallback_summary: str, existing_glossary: Dict[str, str]
    ) -> Tuple[str, Dict[str, str]]:
        if not raw_text or cls.is_refusal(raw_text):
            return fallback_summary, existing_glossary

        cleaned = cls.extract_json_block(raw_text)
        try:
            data = json_repair.loads(cleaned)
            if not isinstance(data, dict):
                return fallback_summary, existing_glossary

            summary = data.get("summary")
            glossary = data.get("new_glossary")

            clean_summary = fallback_summary
            if isinstance(summary, str) and len(summary.strip()) > 10 and not cls.is_refusal(summary):
                clean_summary = summary.strip()

            raw_glossary = glossary if isinstance(glossary, dict) else {}
            updated_glossary = cls.normalize_and_merge_glossary(existing_glossary, raw_glossary)
            return clean_summary, updated_glossary
        except Exception:
            return fallback_summary, existing_glossary


# ==========================================
# 3. КЭШИРОВАНИЕ
# ==========================================

class CacheManager:
    """Управление кэшем на диске."""

    def __init__(self, cache_root: Path, book_id: str):
        self.book_cache_dir = cache_root / book_id
        self.book_cache_dir.mkdir(parents=True, exist_ok=True)

    def _meta_path(self, idx: int) -> Path:
        return self.book_cache_dir / f"{idx:03d}_meta.json"

    def _text_path(self, idx: int) -> Path:
        return self.book_cache_dir / f"{idx:03d}.txt"

    def load_meta(self, idx: int) -> Optional[ChapterMeta]:
        path = self._meta_path(idx)
        if not path.exists():
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
                return ChapterMeta(
                    summary=data.get("summary", ""),
                    glossary=data.get("glossary", {}),
                )
        except Exception as e:
            logger.warning("Ошибка чтения метаданных главы %d: %s", idx, e)
            return None

    def save_meta(self, idx: int, meta: ChapterMeta) -> None:
        path = self._meta_path(idx)
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump({"summary": meta.summary, "glossary": meta.glossary}, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error("Ошибка сохранения метаданных главы %d: %s", idx, e)

    def load_translation(self, idx: int) -> Optional[str]:
        path = self._text_path(idx)
        if path.exists():
            try:
                with open(path, "r", encoding="utf-8") as f:
                    content = f.read().strip()
                if content and not content.startswith("[Ошибка перевода") and not Guardrails.is_refusal(content):
                    return content
                path.unlink(missing_ok=True)
            except Exception as e:
                logger.warning("Ошибка чтения кэша главы %d: %s", idx, e)
        return None

    def save_translation(self, idx: int, content: str) -> None:
        path = self._text_path(idx)
        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write(content)
        except Exception as e:
            logger.error("Ошибка записи перевода главы %d: %s", idx, e)


# ==========================================
# 4. АСИНХРОННЫЙ LLM СЕРВИС (TOKEN PROTECT)
# ==========================================

class AsyncLLMService:
    """Отказоустойчивый клиент с защитой от сжигания токенов при ретраях."""

    def __init__(self, api_key: str, config: PipelineConfig, stats: StatsTracker):
        self.client = AsyncOpenAI(api_key=api_key, base_url=config.base_url)
        self.config = config
        self.stats = stats

    async def call_with_fallback(
        self,
        models_pool: List[str],
        messages: List[Dict[str, str]],
        temperature: float = 0.3,
        is_pass1: bool = False,
    ) -> str:
        last_error: Optional[Exception] = None
        best_candidate: Optional[str] = None

        for model_idx, model in enumerate(models_pool):
            for attempt in range(1, self.config.max_retries_per_model + 1):
                try:
                    extra_body: Dict[str, Any] = {"provider": {"allow_fallbacks": True}}
                    if is_pass1:
                        extra_body["reasoning"] = {"effort": "none"}

                    kwargs: Dict[str, Any] = {
                        "model": model,
                        "messages": messages,
                        "temperature": 0.2 if is_pass1 else temperature,
                        "top_p": 0.95,
                        "timeout": self.config.request_timeout,
                        "extra_body": extra_body,
                    }
                    if is_pass1:
                        kwargs["response_format"] = {"type": "json_object"}

                    response = await self.client.chat.completions.create(**kwargs)
                    if not response.choices or not response.choices[0].message.content:
                        raise ValueError("Модель вернула пустой ответ (NoneType)")

                    content = response.choices[0].message.content.strip()

                    # Сохраняем непустой ответ как резервный
                    if len(content) > 30 and not Guardrails.is_refusal(content):
                        best_candidate = content

                    if Guardrails.is_refusal(content):
                        raise ValueError(f"Отказ модели: {content[:80]}")

                    if not is_pass1 and Guardrails.is_degenerated(content):
                        raise ValueError("Ответ отбракован фильтром дегенерации.")

                    # Считаем токены
                    usage = getattr(response, "usage", None)
                    if usage:
                        p_tok = getattr(usage, "prompt_tokens", 0) or 0
                        c_tok = getattr(usage, "completion_tokens", 0) or 0
                        await self.stats.add_usage(p_tok, c_tok)

                    return content

                except Exception as e:
                    last_error = e
                    jitter = random.uniform(0.5, 2.0)
                    sleep_time = (self.config.backoff_factor ** attempt) + jitter

                    next_model_msg = ""
                    if attempt == self.config.max_retries_per_model and model_idx + 1 < len(models_pool):
                        next_model_msg = f" -> Переключение на модель '{models_pool[model_idx + 1]}'"

                    logger.warning(
                        "[%s] Попытка %d/%d: %s.%s",
                        model, attempt, self.config.max_retries_per_model, str(e)[:90], next_model_msg
                    )
                    await asyncio.sleep(sleep_time)

        # FAIL-SAFE защита от дыр в сюжете
        if best_candidate and len(best_candidate) > 50:
            logger.warning("[Fail-Safe] Использован лучший доступный ответ.")
            return best_candidate

        raise RuntimeError(f"Все модели пула {models_pool} недоступны. Ошибка: {last_error}")


# ==========================================
# 5. СЕРВИС EPUB
# ==========================================

class EpubService:
    """Парсинг и генерация EPUB файлов."""

    EPUB_CSS = """
    body { font-family: serif; margin: 5%; text-align: justify; line-height: 1.45; }
    h2 { text-align: center; margin-top: 1em; margin-bottom: 1.5em; }
    p { text-indent: 1.5em; margin: 0; margin-bottom: 0.3em; }
    em { font-style: italic; }
    strong { font-weight: bold; }
    """

    @classmethod
    def extract_chapters(cls, epub_path: Path, min_word_count: int = 50) -> List[str]:
        book = epub.read_epub(str(epub_path))
        chapters: List[str] = []
        for item in book.get_items():
            if item.get_type() == ebooklib.ITEM_DOCUMENT:
                md_text = TextFormatter.html_to_markdown(item.get_body_content())
                if len(md_text.split()) >= min_word_count:
                    chapters.append(md_text)
        return chapters

    @classmethod
    def build_epub(cls, chapters: List[str], output_path: Path, title: str) -> None:
        book = epub.EpubBook()
        book.set_identifier("ai-literary-pipeline-async-v3.7")
        book.set_title(title)
        book.set_language("ru")
        book.add_author("AI Literary Translator")

        epub_chapters = []
        spine_items = ["nav"]

        for idx, md_text in enumerate(chapters, 1):
            html_body = TextFormatter.markdown_to_html(md_text)
            full_html = f"<h2>Глава {idx}</h2>\n" + html_body

            c = epub.EpubHtml(title=f"Глава {idx}", file_name=f"chapter_{idx:03d}.xhtml", lang="ru")
            c.content = full_html
            book.add_item(c)
            epub_chapters.append(c)
            spine_items.append(c)

        book.toc = tuple(epub_chapters)
        book.add_item(epub.EpubNcx())
        book.add_item(epub.EpubNav())

        nav_css = epub.EpubItem(
            uid="style_nav",
            file_name="style/nav.css",
            media_type="text/css",
            content=cls.EPUB_CSS,
        )
        book.add_item(nav_css)
        book.spine = spine_items

        epub.write_epub(str(output_path), book)


# ==========================================
# 6. КОНВЕЙЕРНЫЙ ПАЙПЛАЙН
# ==========================================

class AsyncTranslationPipeline:
    def __init__(self, api_key: str, config: Optional[PipelineConfig] = None):
        self.config = config or PipelineConfig()
        self.stats = StatsTracker()
        self.llm = AsyncLLMService(api_key=api_key, config=self.config, stats=self.stats)

    async def _analyze_chapter(
        self,
        idx: int,
        text: str,
        prev_summary: str,
        accumulated_glossary: Dict[str, str],
        cache: CacheManager,
    ) -> ChapterMeta:
        cached_meta = cache.load_meta(idx)
        if cached_meta:
            return cached_meta

        analysis_prompt = (
            "You are a professional literary editor and translator. Analyze the text and output strictly valid JSON:\n"
            "{\n"
            '  "summary": "Brief plot summary in Russian (max 120 words)",\n'
            '  "new_glossary": {\n'
            '    "Character Name": "Русский перевод (Пол: М / Ж / Ср)",\n'
            '    "Term/Concept": "Русский перевод"\n'
            "  }\n"
            "}\n"
            "Rules for new_glossary:\n"
            "- Carefully determine character gender by examining English pronouns in context (she/her -> Ж, he/him -> М).\n"
            "- Always specify gender for characters explicitly, e.g.: 'OriginalName': 'РусскоеИмя (Пол: Ж)'.\n"
            "- Use natural Russian phonetics and established translation conventions.\n"
            "- Do not include items already present in Known glossary.\n"
            "- Output raw JSON only."
        )
        user_content = (
            f"Previous summary: {prev_summary}\n"
            f"Known glossary: {json.dumps(accumulated_glossary, ensure_ascii=False)}\n\n"
            f"TEXT:\n{text[:4500]}"
        )

        try:
            raw_res = await self.llm.call_with_fallback(
                models_pool=self.config.models_pass1,
                messages=[
                    {"role": "system", "content": analysis_prompt},
                    {"role": "user", "content": user_content},
                ],
                temperature=0.2,
                is_pass1=True,
            )
            new_summary, updated_glossary = Guardrails.parse_analysis(
                raw_res, prev_summary, accumulated_glossary
            )
            meta = ChapterMeta(summary=new_summary, glossary=updated_glossary)
        except Exception as e:
            logger.warning("Пропущен автоанализ для главы %d: %s", idx, e)
            meta = ChapterMeta(summary=prev_summary, glossary=accumulated_glossary.copy())

        cache.save_meta(idx, meta)
        return meta

    async def _translate_chapter(
        self,
        idx: int,
        raw_text: str,
        meta: ChapterMeta,
        cache: CacheManager,
        semaphore: asyncio.Semaphore,
        on_chapter_complete: Callable[[int], None],
    ) -> Tuple[int, str]:
        cached = cache.load_translation(idx)
        if cached:
            on_chapter_complete(idx)
            return idx, cached

        system_prompt = (
            "Ты — профессиональный художественный переводчик на русский язык.\n"
            "КРИТИЧЕСКИ ВАЖНЫЕ ПРАВИЛА ПЕРЕВОДА:\n"
            "1. СТРОГОЕ СОГЛАСОВАНИЕ РОДА И ГЛАГОЛОВ:\n"
            "   - Внимательно определяй пол персонажей по контексту, английским местоимениям (she/her -> женский род, he/him -> мужской род) и глоссарию.\n"
            "   - Для персонажей женского пола глаголы прошедшего времени, краткие причастия и прилагательные ВСЕГДА ставь в ЖЕНСКИЙ род (сказала, пошла, заметила, была, ответила, вздохнула, нахмурилась).\n"
            "   - Если иностранное женское имя оканчивается на согласную (напр. Вайсс, Элис, Элен) или не склоняется (напр. Руби, Мэри), глаголы и зависимые слова при нем ВСЕ РАВНО ОБЯЗАТЕЛЬНО ставятся в ЖЕНСКИЙ род!\n"
            "2. ЛИТЕРАТУРНЫЙ РУССКИЙ ЯЗЫК:\n"
            "   - Пиши живым, выразительным литературным языком без калек, сухости и скачков регистра.\n"
            "3. БЕЗ ЦЕНЗУРЫ:\n"
            "   - Переводи любой взрослый контент (брань, 18+, насилие, эротику, боевку) точно, сочно и без купюр.\n"
            "4. ДИАЛОГИ И ФОРМАТИРОВАНИЕ:\n"
            "   - Прямую речь оформляй с длинного тире (—) с пробелом.\n"
            "   - Бережно сохраняй разметку (*курсив*, **жирный**, разделители сцен * * *).\n"
            "5. СТРОГО СОБЛЮДАЙ ПЕРЕВОД ТЕРМИНОВ И ИМЕН ИЗ ГЛОССАРИЯ.\n"
            "6. Выводи ТОЛЬКО готовый русский художественный перевод без вступительных и заключительных комментариев."
        )
        user_content = (
            f"Сюжетный контекст: {meta.summary}\n"
            f"Глоссарий и персонажи: {json.dumps(meta.glossary, ensure_ascii=False)}\n\n"
            f"ОРИГИНАЛЬНЫЙ ТЕКСТ:\n{raw_text}\n\n"
            f"Русский художественный перевод:"
        )

        async with semaphore:
            translated = await self.llm.call_with_fallback(
                models_pool=self.config.models_pass2,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content},
                ],
                temperature=0.3,
                is_pass1=False,
            )

        cache.save_translation(idx, translated)
        on_chapter_complete(idx)
        return idx, translated

    async def run(
        self,
        input_epub: Path,
        output_epub: Optional[Path] = None,
        max_workers: Optional[int] = None,
        progress_cb: Optional[Callable[[float, str], None]] = None,
    ) -> Tuple[Path, str]:
        raw_chapters = EpubService.extract_chapters(input_epub)
        total = len(raw_chapters)
        if total == 0:
            raise ValueError("Не удалось извлечь главы из книги.")

        self.stats.total_words_original = sum(len(c.split()) for c in raw_chapters)
        workers_count = max_workers or self.config.max_concurrent_translations
        semaphore = asyncio.Semaphore(workers_count)

        book_id = input_epub.stem
        cache = CacheManager(self.config.cache_dir, book_id)

        logger.info("Запуск конвейера: %d глав, пул: %d воркеров", total, workers_count)

        completed_translations = 0
        analyzed_chapters = 0

        def update_progress(desc: str):
            if progress_cb:
                p = 0.3 * (analyzed_chapters / total) + 0.7 * (completed_translations / total)
                progress_cb(min(p, 1.0), desc)

        def on_chapter_translated(idx: int):
            nonlocal completed_translations
            completed_translations += 1
            logger.info("[✓] Глава %d/%d переведена (%d/%d)", idx, total, completed_translations, total)
            update_progress(f"Переведено глав: {completed_translations}/{total}")

        translation_tasks: List[asyncio.Task] = []
        current_summary = "Начало книги."
        accumulated_glossary: Dict[str, str] = {}

        for idx, text in enumerate(raw_chapters, 1):
            update_progress(f"Анализ сюжета главы {idx}/{total}")

            meta = await self._analyze_chapter(
                idx=idx,
                text=text,
                prev_summary=current_summary,
                accumulated_glossary=accumulated_glossary,
                cache=cache,
            )
            current_summary = meta.summary
            accumulated_glossary = meta.glossary
            analyzed_chapters += 1

            task = asyncio.create_task(
                self._translate_chapter(
                    idx=idx,
                    raw_text=text,
                    meta=meta,
                    cache=cache,
                    semaphore=semaphore,
                    on_chapter_complete=on_chapter_translated,
                )
            )
            translation_tasks.append(task)

        results = await asyncio.gather(*translation_tasks, return_exceptions=True)

        translated_chapters: List[str] = []
        for i, res in enumerate(results, 1):
            if isinstance(res, Exception):
                logger.error("Критический сбой перевода главы %d: %s", i, res)
                translated_chapters.append(f"[Ошибка перевода главы {i}: {res}]")
            else:
                _, text = res
                translated_chapters.append(text)

        if output_epub is None:
            output_epub = input_epub.parent / f"{book_id}_RU.epub"

        EpubService.build_epub(
            chapters=translated_chapters,
            output_path=output_epub,
            title=f"{book_id} (RU)",
        )

        stats_summary = self.stats.get_summary()
        logger.info("Готово! %s", stats_summary)

        if progress_cb:
            progress_cb(1.0, f"Готово! {stats_summary}")

        return output_epub, stats_summary


# ==========================================
# 7. ИНТЕРФЕЙСЫ (CLI И GRADIO)
# ==========================================

def parse_model_list(models_str: str, default_list: List[str]) -> List[str]:
    """Парсит список моделей через запятую."""
    if not models_str or not models_str.strip():
        return default_list
    models = [m.strip() for m in models_str.split(",") if m.strip()]
    return models if models else default_list


def launch_web_gui():
    import gradio as gr

    async def web_process(
        epub_file, api_key: str, workers: int, models_p1_raw: str, models_p2_raw: str, progress=gr.Progress()
    ):
        if not epub_file:
            return None, "Загрузите .epub файл."
        if not api_key:
            return None, "Введите OpenRouter API Key."

        p1_models = parse_model_list(models_p1_raw, ["deepseek/deepseek-v3.2"])
        p2_models = parse_model_list(models_p2_raw, ["deepseek/deepseek-v3.2"])

        cfg = PipelineConfig(
            max_concurrent_translations=int(workers),
            models_pass1=p1_models,
            models_pass2=p2_models,
        )
        pipeline = AsyncTranslationPipeline(api_key=api_key, config=cfg)

        def cb(frac, desc):
            progress(frac, desc=desc)

        try:
            out_path, stats = await pipeline.run(
                input_epub=Path(epub_file.name),
                max_workers=int(workers),
                progress_cb=cb,
            )
            return str(out_path), f"Перевод успешно завершен!\n\n{stats}"
        except Exception as e:
            logger.exception("Ошибка пайплайна")
            return None, f"Ошибка пайплайна: {e}"

    with gr.Blocks(title="AI Literary Novel Translator") as ui:
        gr.Markdown("## ⚡ Fast Async AI Literary Translator (Stable Engine v3.7)")
        with gr.Row():
            with gr.Column():
                f_in = gr.File(label="Исходный EPUB", file_types=[".epub"])
                k_in = gr.Textbox(
                    label="OpenRouter API Key",
                    type="password",
                    value=os.getenv("OPENROUTER_API_KEY", ""),
                    placeholder="sk-or-v1-...",
                )
                w_in = gr.Slider(
                    minimum=2, maximum=30, value=15, step=1, label="Параллельных глав (Concurrency)"
                )
                m1_in = gr.Textbox(
                    label="Модели Фазы 1 (Анализ сюжета и глоссарий, через запятую)",
                    value="deepseek/deepseek-v3.2",
                )
                m2_in = gr.Textbox(
                    label="Модели Фазы 2 (Художественный перевод, через запятую)",
                    value="deepseek/deepseek-v3.2",
                )
                btn = gr.Button("Начать перевод", variant="primary")
            with gr.Column():
                st = gr.Textbox(label="Статус и статистика", interactive=False, lines=5)
                f_out = gr.File(label="Готовая книга (.epub)")

        btn.click(web_process, inputs=[f_in, k_in, w_in, m1_in, m2_in], outputs=[f_out, st])

    ui.launch(inbrowser=True)


def main():
    parser = argparse.ArgumentParser(description="Async Fast AI Literary Translator v3.7")
    parser.add_argument("--file", type=Path, help="Входной .epub файл")
    parser.add_argument("--key", default=os.getenv("OPENROUTER_API_KEY"), help="OpenRouter API Key")
    parser.add_argument("--workers", type=int, default=15, help="Параллельных глав (по умолчанию 15)")
    parser.add_argument("--p1-models", default="", help="Пул моделей Фазы 1 через запятую")
    parser.add_argument("--p2-models", default="", help="Пул моделей Фазы 2 через запятую")
    parser.add_argument("--gui", action="store_true", help="Запустить Web GUI")
    args = parser.parse_args()

    if args.gui or not args.file:
        launch_web_gui()
    else:
        if not args.key:
            logger.error("Укажите API-ключ через --key или переменную OPENROUTER_API_KEY.")
            return

        cfg = PipelineConfig(
            max_concurrent_translations=args.workers,
            models_pass1=parse_model_list(args.p1_models, ["deepseek/deepseek-v3.2"]),
            models_pass2=parse_model_list(args.p2_models, ["deepseek/deepseek-v3.2"]),
        )
        pipeline = AsyncTranslationPipeline(api_key=args.key, config=cfg)

        def cli_cb(frac, desc):
            print(f"[{int(frac * 100):02d}%] {desc}")

        out, stats = asyncio.run(
            pipeline.run(
                input_epub=args.file,
                max_workers=args.workers,
                progress_cb=cli_cb,
            )
        )
        print(f"\n[✓] Перевод завершен: {out}\n{stats}")


if __name__ == "__main__":
    main()