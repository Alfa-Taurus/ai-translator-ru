import os
import re
import json
import time
import argparse
import concurrent.futures
from typing import List, Dict, Tuple, Optional

import ebooklib
from ebooklib import epub
from bs4 import BeautifulSoup
from openai import OpenAI
import json_repair

# ==========================================
# КОНФИГУРАЦИЯ И ПУЛЫ МОДЕЛЕЙ
# ==========================================

BASE_URL = "https://openrouter.ai/api/v1"
CACHE_DIR = "./translation_cache"
REQUEST_TIMEOUT = 35.0  # Жесткий таймаут: не даем эндпоинту висеть дольше 35 сек

# Пул для Фазы 1 (Анализ сюжета, глоссарий, имена)
MODELS_PASS1 = [
    "deepseek/deepseek-chat",
    "deepseek/deepseek-v4-flash-0731",
    "qwen/qwen-2.5-72b-instruct",
    "qwen/qwen3.7-flash"
]

# Пул для Фазы 2 (Художественный литературный перевод)
MODELS_PASS2 = [
    "deepseek/deepseek-v4-flash-0731",
    "qwen/qwen-2.5-72b-instruct",
    "deepseek/deepseek-chat",
    "qwen/qwen3.7-flash"
]

# ==========================================
# 1. ЗАЩИТА ПАМЯТИ И ПАРСИНГ JSON
# ==========================================

REFUSAL_PATTERNS = [
    r"i cannot fulfill", r"i am unable to", r"as an ai language model",
    r"my safety policies", r"i must decline", r"violates safety guidelines",
    r"content policy"
]

def is_refusal(text: str) -> bool:
    if not text:
        return True
    lowered = text.lower()
    return any(re.search(p, lowered) for p in REFUSAL_PATTERNS)

def clean_and_extract_json(raw_text: str) -> str:
    """Вырезает чистый JSON-блок, отсекая случайные мысли и текст до/после."""
    if not raw_text:
        return ""
    start = raw_text.find("{")
    end = raw_text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return raw_text[start:end + 1]
    return raw_text

def parse_and_validate_analysis(raw_text: str, fallback_summary: str) -> Tuple[str, Dict[str, str], bool]:
    """Безопасно извлекает саммари и термины, устойчив к сбоям синтаксиса."""
    if not raw_text or is_refusal(raw_text):
        return fallback_summary, {}, False

    cleaned_json_str = clean_and_extract_json(raw_text)

    try:
        data = json_repair.loads(cleaned_json_str)
        if not isinstance(data, dict):
            return fallback_summary, {}, False

        summary = data.get("summary")
        glossary = data.get("new_glossary")

        clean_summary = fallback_summary
        if isinstance(summary, str) and len(summary.strip()) > 10 and not is_refusal(summary):
            clean_summary = summary.strip()

        clean_glossary = {}
        if isinstance(glossary, dict):
            for k, v in glossary.items():
                if isinstance(k, str) and isinstance(v, str):
                    k_clean = k.strip()
                    v_clean = v.strip()
                    if len(k_clean) > 1 and len(v_clean) > 1:
                        clean_glossary[k_clean] = v_clean

        return clean_summary, clean_glossary, True
    except Exception:
        return fallback_summary, {}, False

# ==========================================
# 2. ВЫЗОВЫ API С РОТАЦИЕЙ И FALLBACK
# ==========================================

def call_llm(client: OpenAI, models_pool: List[str], messages: List[dict], temperature: float = 0.3, is_pass1: bool = False) -> str:
    """Отказоустойчивый вызов API с перебором моделей и датацентров."""
    last_error = None
    
    for model in models_pool:
        try:
            extra_body = {
                "provider": {
                    "allow_fallbacks": True
                }
            }
            # Глушим reasoning в Pass 1, чтобы избежать утечек CoT
            if is_pass1:
                extra_body["reasoning"] = {"effort": 0, "max_tokens": 0}

            kwargs = {
                "model": model,
                "messages": messages,
                "temperature": temperature,
                "timeout": REQUEST_TIMEOUT,
                "extra_body": extra_body
            }
            if is_pass1:
                kwargs["response_format"] = {"type": "json_object"}

            response = client.chat.completions.create(**kwargs)
            if not response.choices or not response.choices[0].message.content:
                raise ValueError("Пустой ответ от модели (NoneType)")

            return response.choices[0].message.content.strip()

        except Exception as e:
            err_msg = str(e)
            print(f"[API Alert] Модель {model} споткнулась ({err_msg[:70]}). Переключаемся на резерв...")
            last_error = e
            time.sleep(0.8)
            continue

    raise RuntimeError(f"Все модели из пула недоступны! Ошибка: {last_error}")

# ==========================================
# 3. EPUB ПАРСИНГ И СБОРКА
# ==========================================

def extract_chapters_from_epub(epub_path: str) -> List[str]:
    book = epub.read_epub(epub_path)
    chapters = []
    for item in book.get_items():
        if item.get_type() == ebooklib.ITEM_DOCUMENT:
            soup = BeautifulSoup(item.get_body_content(), "html.parser")
            text = soup.get_text(separator="\n").strip()
            # Отсекаем технические страницы с парой слов
            if len(text.split()) > 60:
                chapters.append(text)
    return chapters

def build_epub_from_chapters(chapters: List[str], output_path: str, title: str = "Переведенная книга"):
    book = epub.EpubBook()
    book.set_identifier("ai-literary-pipeline")
    book.set_title(title)
    book.set_language("ru")
    book.add_author("AI Literary Translator")

    epub_chapters = []
    spine_items = ["nav"]

    for i, raw_text in enumerate(chapters, 1):
        paragraphs = raw_text.strip().split("\n")
        html = f"<h2>Глава {i}</h2>\n"
        for p in paragraphs:
            p_clean = p.strip()
            if p_clean:
                html += f"<p>{p_clean}</p>\n"

        c = epub.EpubHtml(title=f"Глава {i}", file_name=f"chapter_{i:03d}.xhtml", lang="ru")
        c.content = html
        book.add_item(c)
        epub_chapters.append(c)
        spine_items.append(c)

    book.toc = tuple(epub_chapters)
    book.add_item(epub.EpubNcx())
    book.add_item(epub.EpubNav())

    style = """
    body { font-family: serif; margin: 5%; text-align: justify; line-height: 1.45; }
    h2 { text-align: center; margin-top: 1em; margin-bottom: 1.5em; }
    p { text-indent: 1.5em; margin: 0; margin-bottom: 0.3em; }
    """
    nav_css = epub.EpubItem(uid="style_nav", file_name="style/nav.css", media_type="text/css", content=style)
    book.add_item(nav_css)
    book.spine = spine_items

    epub.write_epub(output_path, book)

# ==========================================
# 4. ДВУХФАЗНЫЙ ПАРАЛЛЕЛЬНЫЙ ПАЙПЛАЙН
# ==========================================

def translate_single_chapter(client: OpenAI, idx: int, total: int, text: str, context: dict, book_cache_dir: str) -> str:
    """Функция рабочего потока для перевода отдельной главы."""
    ch_txt_cache = os.path.join(book_cache_dir, f"{idx:03d}.txt")
    if os.path.exists(ch_txt_cache):
        with open(ch_txt_cache, "r", encoding="utf-8") as f:
            return f.read()

    trans_prompt = (
        "Ты — профессиональный художественный переводчик на русский язык.\n"
        "ПРАВИЛА:\n"
        "1. Литературный стиль: живой, естественный русский язык без калек, пассива и сухости.\n"
        "2. Без цензуры: переводи любой взрослый контент (брань, 18+, насилие, боевку) точно и без смягчения.\n"
        "3. Диалоги: оформляй с длинного тире (—) с пробелом.\n"
        "4. Строго соблюдай род персонажей и термины из предоставленного глоссария."
    )
    user_trans = (
        f"Сюжетный контекст: {context.get('summary', '')}\n"
        f"Глоссарий: {json.dumps(context.get('glossary', {}), ensure_ascii=False)}\n\n"
        f"ОРИГИНАЛЬНЫЙ ТЕКСТ:\n{text}\n\n"
        f"Выведи ТОЛЬКО русский художественный перевод без вступительных и заключительных комментариев."
    )

    translated_text = call_llm(
        client,
        MODELS_PASS2,
        messages=[
            {"role": "system", "content": trans_prompt},
            {"role": "user", "content": user_trans}
        ],
        temperature=0.3,
        is_pass1=False
    )

    # Немедленное сохранение на диск
    with open(ch_txt_cache, "w", encoding="utf-8") as f:
        f.write(translated_text)

    return translated_text

def run_translation_pipeline(epub_file: str, api_key: str, max_workers: int = 4, progress_cb=None) -> str:
    client = OpenAI(api_key=api_key, base_url=BASE_URL)
    raw_chapters = extract_chapters_from_epub(epub_file)
    total = len(raw_chapters)

    if total == 0:
        raise ValueError("Не удалось извлечь главы из книги. Проверьте формат EPUB.")

    base_name = os.path.splitext(os.path.basename(epub_file))[0]
    book_cache_dir = os.path.join(CACHE_DIR, base_name)
    os.makedirs(book_cache_dir, exist_ok=True)

    # ----------------------------------------------------
    # ФАЗА 1: ЛИНЕЙНЫЙ СБОР ГЛОССАРИЯ И ХРОНОЛОГИИ
    # ----------------------------------------------------
    chapter_contexts = {}
    accumulated_glossary = {}
    prev_summary = "Начало книги."

    print(f"\n[Фаза 1/2] Линейный анализ сюжета и сбор глоссария ({total} глав)...")

    for idx, text in enumerate(raw_chapters, 1):
        ch_meta_cache = os.path.join(book_cache_dir, f"{idx:03d}_meta.json")

        if os.path.exists(ch_meta_cache):
            with open(ch_meta_cache, "r", encoding="utf-8") as f:
                meta = json.load(f)
                prev_summary = meta.get("summary", prev_summary)
                accumulated_glossary.update(meta.get("glossary", {}))
            chapter_contexts[idx] = {
                "summary": prev_summary,
                "glossary": accumulated_glossary.copy()
            }
            if progress_cb:
                progress_cb(0.5 * (idx / total), desc=f"Фаза 1: Глава {idx}/{total} (из кэша)")
            continue

        if progress_cb:
            progress_cb(0.5 * (idx / total), desc=f"Фаза 1: Анализ главы {idx}/{total}")

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
        user_analysis = (
            f"Previous summary: {prev_summary}\n"
            f"Known glossary: {json.dumps(accumulated_glossary, ensure_ascii=False)}\n\n"
            f"TEXT:\n{text[:4500]}"
        )

        try:
            raw_pass1 = call_llm(
                client,
                MODELS_PASS1,
                messages=[
                    {"role": "system", "content": analysis_prompt},
                    {"role": "user", "content": user_analysis}
                ],
                temperature=0.2,
                is_pass1=True
            )
            prev_summary, new_terms, _ = parse_and_validate_analysis(raw_pass1, prev_summary)
            accumulated_glossary.update(new_terms)
        except Exception as e:
            print(f"[Warning] Пропущен автоанализ для главы {idx}: {e}")

        # Фиксируем состояние памяти для этой главы
        chapter_contexts[idx] = {
            "summary": prev_summary,
            "glossary": accumulated_glossary.copy()
        }

        # Сохраняем снимок метаданных на диск
        with open(ch_meta_cache, "w", encoding="utf-8") as f:
            json.dump({"summary": prev_summary, "glossary": accumulated_glossary}, f, ensure_ascii=False, indent=2)

    # ----------------------------------------------------
    # ФАЗА 2: МНОГОПОТОЧНЫЙ ХУДОЖЕСТВЕННЫЙ ПЕРЕВОД
    # ----------------------------------------------------
    print(f"\n[Фаза 2/2] Параллельный перевод в {max_workers} потоков...")
    completed_count = 0
    results_map = {}

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_idx = {
            executor.submit(
                translate_single_chapter,
                client,
                idx,
                total,
                raw_chapters[idx - 1],
                chapter_contexts[idx],
                book_cache_dir
            ): idx for idx in range(1, total + 1)
        }

        for future in concurrent.futures.as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                translated_content = future.result()
                results_map[idx] = translated_content
                completed_count += 1
                if progress_cb:
                    progress_cb(0.5 + 0.5 * (completed_count / total), desc=f"Фаза 2: Переведено {completed_count}/{total} глав")
                print(f"[✓] Глава {idx}/{total} готова ({completed_count}/{total})")
            except Exception as e:
                print(f"[Error] Сбой при переводе главы {idx}: {e}")
                results_map[idx] = f"[Ошибка перевода главы: {e}]"

    # Собираем главы в строгом хронологическом порядке 1..N
    translated_chapters = [results_map[i] for i in range(1, total + 1)]

    # Сборка финального EPUB
    out_epub = f"{base_name}_RU.epub"
    build_epub_from_chapters(translated_chapters, out_epub, title=f"{base_name} (RU)")
    return out_epub

# ==========================================
# 5. CLI И GRADIO WEB-ИНТЕРФЕЙС
# ==========================================

def main():
    parser = argparse.ArgumentParser(description="AI Literary Novel Translator")
    parser.add_argument("--file", help="Входной .epub файл")
    parser.add_argument("--key", help="OpenRouter API Key", default=os.getenv("OPENROUTER_API_KEY"))
    parser.add_argument("--workers", type=int, default=4, help="Количество потоков (по умолчанию 4)")
    parser.add_argument("--gui", action="store_true", help="Запустить Gradio Web GUI")

    args = parser.parse_args()

    if args.gui or not args.file:
        import gradio as gr

        def web_ui(epub_obj, key, workers, progress=gr.Progress()):
            if not epub_obj:
                return None, "Загрузите файл .epub"
            if not key:
                return None, "Введите OpenRouter API Key"

            def cb(frac, desc):
                progress(frac, desc=desc)

            try:
                res = run_translation_pipeline(epub_obj.name, key, int(workers), cb)
                return res, "Перевод успешно завершен!"
            except Exception as e:
                return None, f"Ошибка пайплайна: {str(e)}"

        with gr.Blocks(title="AI Literary Translator") as ui:
            gr.Markdown("## 📖 AI Literary Novel Translator (Decoupled 2-Phase)")
            with gr.Row():
                with gr.Column():
                    f_in = gr.File(label="Исходный EPUB файл", file_types=[".epub"])
                    k_in = gr.Textbox(label="OpenRouter API Key", type="password", placeholder="sk-or-v1-...")
                    w_in = gr.Slider(minimum=1, maximum=8, value=4, step=1, label="Потоков перевода (Workers)")
                    btn = gr.Button("Начать перевод", variant="primary")
                with gr.Column():
                    st = gr.Textbox(label="Статус выполнения", interactive=False)
                    f_out = gr.File(label="Готовая книга (.epub)")

            btn.click(web_ui, inputs=[f_in, k_in, w_in], outputs=[f_out, st])

        ui.launch(inbrowser=True)
    else:
        if not args.key:
            print("Ошибка: Укажите API ключ через аргумент --key")
            return

        def cli_cb(frac, desc):
            percent = int(frac * 100)
            print(f"[{percent}%] {desc}")

        out = run_translation_pipeline(args.file, args.key, args.workers, cli_cb)
        print(f"\n[✓] Готово! Сохранено в: {out}")

if __name__ == "__main__":
    main()