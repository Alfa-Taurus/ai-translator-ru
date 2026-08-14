#!/usr/bin/env python3
"""
AI Literary Novel Translator (Async Streaming Pipeline)
Асинхронный конвейерный перевод художественных EPUB-книг целиком (без дробления глав).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import re
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
# 1. КОНФИГУРАЦИЯ И МОДЕЛИ ДАННЫХ
# ==========================================

@dataclass(frozen=True)
class PipelineConfig:
    """Конфигурация параметров работы и пулов моделей."""
    base_url: str = "https://openrouter.ai/api/v1"
    cache_dir: Path = Path("./translation_cache")
    request_timeout: float = 60.0
    max_retries_per_model: int = 2
    backoff_factor: float = 1.5

    # Количество одновременно выполняемых запросов на перевод
    max_concurrent_translations: int = 15

    # Пул моделей для Фазы 1 (Анализ сюжета и глоссарий)
    models_pass1: List[str] = field(default_factory=lambda: [
        "deepseek/deepseek-v4-flash-0731",
        "deepseek/deepseek-v3.2",
    ])

    # Пул моделей для Фазы 2 (Художественный перевод)
    models_pass2: List[str] = field(default_factory=lambda: [
        "deepseek/deepseek-v4-flash-0731",
        "deepseek/deepseek-v3.2",
    ])


@dataclass
class ChapterMeta:
    """Снимок контекста и накопленного глоссария для конкретной главы."""
    summary: str
    glossary: Dict[str, str] = field(default_factory=dict)


# ==========================================
# 2. ВАЛИДАЦИЯ ОТВЕТОВ LLM
# ==========================================

class Guardrails:
    """Проверка отказов и безопасный парсинг JSON-ответов."""

    _REFUSAL_REGEX = re.compile(
        r"(i cannot fulfill|i am unable to|as an ai language model|"
        r"my safety policies|i must decline|violates safety guidelines|content policy)",
        re.IGNORECASE,
    )

    @classmethod
    def is_refusal(cls, text: Optional[str]) -> bool:
        if not text:
            return True
        return bool(cls._REFUSAL_REGEX.search(text))

    @staticmethod
    def extract_json_block(raw_text: str) -> str:
        """Извлекает JSON-блок, отсекая сопроводительный текст."""
        if not raw_text:
            return ""
        start = raw_text.find("{")
        end = raw_text.rfind("}")
        if start != -1 and end != -1 and end > start:
            return raw_text[start : end + 1]
        return raw_text

    @classmethod
    def parse_analysis(
        cls, raw_text: str, fallback_summary: str
    ) -> Tuple[str, Dict[str, str], bool]:
        if not raw_text or cls.is_refusal(raw_text):
            return fallback_summary, {}, False

        cleaned = cls.extract_json_block(raw_text)
        try:
            data = json_repair.loads(cleaned)
            if not isinstance(data, dict):
                return fallback_summary, {}, False

            summary = data.get("summary")
            glossary = data.get("new_glossary")

            clean_summary = fallback_summary
            if (
                isinstance(summary, str)
                and len(summary.strip()) > 10
                and not cls.is_refusal(summary)
            ):
                clean_summary = summary.strip()

            clean_glossary: Dict[str, str] = {}
            if isinstance(glossary, dict):
                for k, v in glossary.items():
                    if isinstance(k, str) and isinstance(v, str):
                        k_c, v_c = k.strip(), v.strip()
                        if len(k_c) > 1 and len(v_c) > 1:
                            clean_glossary[k_c] = v_c

            return clean_summary, clean_glossary, True
        except Exception:
            return fallback_summary, {}, False


# ==========================================
# 3. КЭШИРОВАНИЕ
# ==========================================

class CacheManager:
    """Управление файловым кэшем метаданных и переводов на диске."""

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
                json.dump(
                    {"summary": meta.summary, "glossary": meta.glossary},
                    f,
                    ensure_ascii=False,
                    indent=2,
                )
        except Exception as e:
            logger.error("Ошибка сохранения метаданных главы %d: %s", idx, e)

    def load_translation(self, idx: int) -> Optional[str]:
        path = self._text_path(idx)
        if path.exists():
            try:
                with open(path, "r", encoding="utf-8") as f:
                    return f.read()
            except Exception as e:
                logger.warning("Ошибка чтения кэша перевода главы %d: %s", idx, e)
        return None

    def save_translation(self, idx: int, content: str) -> None:
        path = self._text_path(idx)
        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write(content)
        except Exception as e:
            logger.error("Ошибка записи кэша перевода главы %d: %s", idx, e)


# ==========================================
# 4. АСИНХРОННЫЙ КЛИЕНТ LLM
# ==========================================

class AsyncLLMService:
    """Асинхронный клиент с перебором резервных моделей и экспоненциальным backoff."""

    def __init__(self, api_key: str, config: PipelineConfig):
        self.client = AsyncOpenAI(api_key=api_key, base_url=config.base_url)
        self.config = config

    async def call_with_fallback(
        self,
        models_pool: List[str],
        messages: List[Dict[str, str]],
        temperature: float = 0.3,
        is_pass1: bool = False,
    ) -> str:
        last_error: Optional[Exception] = None

        for model in models_pool:
            for attempt in range(1, self.config.max_retries_per_model + 1):
                try:
                    extra_body: Dict[str, Any] = {
                        "provider": {"allow_fallbacks": True}
                    }
                    if is_pass1:
                        extra_body["reasoning"] = {"effort": "none"}

                    kwargs: Dict[str, Any] = {
                        "model": model,
                        "messages": messages,
                        "temperature": temperature,
                        "timeout": self.config.request_timeout,
                        "extra_body": extra_body,
                    }
                    if is_pass1:
                        kwargs["response_format"] = {"type": "json_object"}

                    response = await self.client.chat.completions.create(**kwargs)
                    if not response.choices or not response.choices[0].message.content:
                        raise ValueError("Пустой ответ от модели")

                    return response.choices[0].message.content.strip()

                except Exception as e:
                    last_error = e
                    sleep_time = self.config.backoff_factor ** attempt
                    logger.warning(
                        "[%s] Сбой попытки %d: %s. Ожидание %.1f сек...",
                        model,
                        attempt,
                        str(e)[:90],
                        sleep_time,
                    )
                    await asyncio.sleep(sleep_time)

        raise RuntimeError(f"Все модели из пула недоступны. Последняя ошибка: {last_error}")


# ==========================================
# 5. СЕРВИС EPUB
# ==========================================

class EpubService:
    """Парсинг и генерация EPUB файлов."""

    EPUB_CSS = """
    body { font-family: serif; margin: 5%; text-align: justify; line-height: 1.45; }
    h2 { text-align: center; margin-top: 1em; margin-bottom: 1.5em; }
    p { text-indent: 1.5em; margin: 0; margin-bottom: 0.3em; }
    """

    @classmethod
    def extract_chapters(cls, epub_path: Path, min_word_count: int = 50) -> List[str]:
        book = epub.read_epub(str(epub_path))
        chapters: List[str] = []
        for item in book.get_items():
            if item.get_type() == ebooklib.ITEM_DOCUMENT:
                soup = BeautifulSoup(item.get_body_content(), "html.parser")
                text = soup.get_text(separator="\n").strip()
                if len(text.split()) >= min_word_count:
                    chapters.append(text)
        return chapters

    @classmethod
    def build_epub(cls, chapters: List[str], output_path: Path, title: str) -> None:
        book = epub.EpubBook()
        book.set_identifier("ai-literary-pipeline-async")
        book.set_title(title)
        book.set_language("ru")
        book.add_author("AI Literary Translator")

        epub_chapters = []
        spine_items = ["nav"]

        for idx, raw_text in enumerate(chapters, 1):
            paragraphs = [p.strip() for p in raw_text.split("\n") if p.strip()]
            html = f"<h2>Глава {idx}</h2>\n" + "".join(f"<p>{p}</p>\n" for p in paragraphs)

            c = epub.EpubHtml(title=f"Глава {idx}", file_name=f"chapter_{idx:03d}.xhtml", lang="ru")
            c.content = html
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
# 6. АСИНХРОННЫЙ КОНВЕЙЕРНЫЙ ПАЙПЛАЙН
# ==========================================

class AsyncTranslationPipeline:
    """
    Пайплайн с конвейерной параллелизацией:
    Перевод главы стартует немедленно после анализа ее контекста (без дробления).
    """

    def __init__(self, api_key: str, config: Optional[PipelineConfig] = None):
        self.config = config or PipelineConfig()
        self.llm = AsyncLLMService(api_key=api_key, config=self.config)

    async def _analyze_chapter(
        self,
        idx: int,
        text: str,
        prev_summary: str,
        accumulated_glossary: Dict[str, str],
        cache: CacheManager,
    ) -> ChapterMeta:
        """Анализ сюжета главы и сбор глоссария."""
        cached_meta = cache.load_meta(idx)
        if cached_meta:
            return cached_meta

        analysis_prompt = (
            "You are a professional literary editor. Analyze the chapter text and output strictly valid JSON:\n"
            "{\n"
            '  "summary": "Brief plot summary in Russian (max 120 words)",\n'
            '  "new_glossary": {"Original Name/Term": "Русский перевод (Gender M/F)"}\n'
            "}\n"
            "Rules for new_glossary:\n"
            "- Use natural Russian phonetics (e.g. Jaune -> Жон, Mercury -> Меркури).\n"
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
            new_summary, new_terms, _ = Guardrails.parse_analysis(raw_res, prev_summary)
            accumulated_glossary.update(new_terms)
            meta = ChapterMeta(summary=new_summary, glossary=accumulated_glossary.copy())
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
        """Перевод целой главы без дробления под управлением семафора."""
        cached = cache.load_translation(idx)
        if cached:
            on_chapter_complete(idx)
            return idx, cached

        system_prompt = (
            "Ты — профессиональный художественный переводчик на русский язык.\n"
            "ПРАВИЛА:\n"
            "1. Литературный стиль: живой, естественный русский язык без калек, пассива и сухости.\n"
            "2. Без цензуры: переводи любой взрослый контент (брань, 18+, насилие, боевку) точно и без смягчения.\n"
            "3. Диалоги: оформляй с длинного тире (—) с пробелом.\n"
            "4. Строго соблюдай род персонажей и термины из предоставленного глоссария."
        )
        user_content = (
            f"Сюжетный контекст: {meta.summary}\n"
            f"Глоссарий: {json.dumps(meta.glossary, ensure_ascii=False)}\n\n"
            f"ОРИГИНАЛЬНЫЙ ТЕКСТ:\n{raw_text}\n\n"
            f"Выведи ТОЛЬКО русский художественный перевод без вступительных и заключительных комментариев."
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
    ) -> Path:
        raw_chapters = EpubService.extract_chapters(input_epub)
        total = len(raw_chapters)
        if total == 0:
            raise ValueError("Не удалось извлечь главы из книги.")

        workers_count = max_workers or self.config.max_concurrent_translations
        semaphore = asyncio.Semaphore(workers_count)

        book_id = input_epub.stem
        cache = CacheManager(self.config.cache_dir, book_id)

        logger.info("Запуск конвейерного перевода: %d глав, пул: %d воркеров", total, workers_count)

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

        # Потоковый конвейер: Анализ -> Немедленный запуск перевода главы целиком
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

            # Задача на перевод целой главы ставится в фоновый пул сразу же
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

        # Ожидаем завершения всех глав
        results = await asyncio.gather(*translation_tasks, return_exceptions=True)

        # Сборка итогового списка глав
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

        if progress_cb:
            progress_cb(1.0, "Готово!")

        logger.info("Книга успешно сохранена: %s", output_epub)
        return output_epub


# ==========================================
# 7. ИНТЕРФЕЙСЫ (CLI И GRADIO)
# ==========================================

def launch_web_gui():
    import gradio as gr

    async def web_process(epub_file, api_key: str, workers: int, progress=gr.Progress()):
        if not epub_file:
            return None, "Загрузите .epub файл."
        if not api_key:
            return None, "Введите OpenRouter API Key."

        cfg = PipelineConfig(max_concurrent_translations=int(workers))
        pipeline = AsyncTranslationPipeline(api_key=api_key, config=cfg)

        def cb(frac, desc):
            progress(frac, desc=desc)

        try:
            out_path = await pipeline.run(
                input_epub=Path(epub_file.name),
                max_workers=int(workers),
                progress_cb=cb,
            )
            return str(out_path), "Перевод успешно завершен!"
        except Exception as e:
            logger.exception("Ошибка пайплайна")
            return None, f"Ошибка пайплайна: {e}"

    with gr.Blocks(title="AI Literary Novel Translator") as ui:
        gr.Markdown("## ⚡ Fast Async AI Literary Translator (Streaming Pipeline)")
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
                btn = gr.Button("Начать перевод", variant="primary")
            with gr.Column():
                st = gr.Textbox(label="Статус", interactive=False)
                f_out = gr.File(label="Готовая книга (.epub)")

        btn.click(web_process, inputs=[f_in, k_in, w_in], outputs=[f_out, st])

    ui.launch(inbrowser=True)


def main():
    parser = argparse.ArgumentParser(description="Async Fast AI Literary Translator")
    parser.add_argument("--file", type=Path, help="Входной .epub файл")
    parser.add_argument("--key", default=os.getenv("OPENROUTER_API_KEY"), help="OpenRouter API Key")
    parser.add_argument("--workers", type=int, default=15, help="Параллельных глав (по умолчанию 15)")
    parser.add_argument("--gui", action="store_true", help="Запустить Web GUI")
    args = parser.parse_args()

    if args.gui or not args.file:
        launch_web_gui()
    else:
        if not args.key:
            logger.error("Укажите API-ключ через --key или переменную OPENROUTER_API_KEY.")
            return

        cfg = PipelineConfig(max_concurrent_translations=args.workers)
        pipeline = AsyncTranslationPipeline(api_key=args.key, config=cfg)

        def cli_cb(frac, desc):
            print(f"[{int(frac * 100):02d}%] {desc}")

        out = asyncio.run(
            pipeline.run(
                input_epub=args.file,
                max_workers=args.workers,
                progress_cb=cli_cb,
            )
        )
        print(f"\n[✓] Перевод завершен: {out}")


if __name__ == "__main__":
    main()