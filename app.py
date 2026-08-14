import os
import json
import argparse
import ebooklib
from ebooklib import epub
from bs4 import BeautifulSoup
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

DEFAULT_MODEL = "qwen/qwen3.7-flash"
BASE_URL = "https://openrouter.ai/api/v1"

# ==========================================
# 1. РАБОТА С EPUB (ЧТЕНИЕ И СБОРКА)
# ==========================================

def extract_chapters_from_epub(epub_path):
    """Извлекает главы из EPUB в виде чистого текста."""
    book = epub.read_epub(epub_path)
    chapters = []
    for item in book.get_items():
        if item.get_type() == ebooklib.ITEM_DOCUMENT:
            soup = BeautifulSoup(item.get_body_content(), "html.parser")
            text = soup.get_text(separator="\n").strip()
            # Пропускаем пустые и сугубо технические разметки
            if len(text.split()) > 60:
                chapters.append(text)
    return chapters

def build_epub_from_chapters(chapters, output_path, title="Переведенная книга"):
    """Собирает список переведенных глав обратно в валидный EPUB с форматированием."""
    book = epub.EpubBook()
    book.set_identifier("ai-pipeline-novel-ru")
    book.set_title(title)
    book.set_language("ru")
    book.add_author("AI Novel Translator")

    epub_chapters = []
    spine_items = ["nav"]

    for i, raw_text in enumerate(chapters, 1):
        paragraphs = raw_text.strip().split("\n")
        html = f"<h2>Глава {i}</h2>\n"
        for p in paragraphs:
            p_clean = p.strip()
            if p_clean:
                html += f"<p>{p_clean}</p>\n"

        c = epub.EpubHtml(title=f"Глава {i}", file_name=f"chap_{i:03d}.xhtml", lang="ru")
        c.content = html
        book.add_item(c)
        epub_chapters.append(c)
        spine_items.append(c)

    book.toc = tuple(epub_chapters)
    book.add_item(epub.EpubNcx())
    book.add_item(epub.EpubNav())

    style = """
    body { font-family: serif; margin: 5%; text-align: justify; line-height: 1.4; }
    h2 { text-align: center; margin-bottom: 1.5em; }
    p { text-indent: 1.5em; margin: 0; }
    """
    nav_css = epub.EpubItem(uid="style_nav", file_name="style/nav.css", media_type="text/css", content=style)
    book.add_item(nav_css)
    book.spine = spine_items

    epub.write_epub(output_path, book)

# ==========================================
# 2. УСТОЙЧИВЫЙ КЛИЕНТ API (AUTO-RETRY)
# ==========================================

@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=2, max=20),
    reraise=True
)
def safe_chat_completion(client, **kwargs):
    """Вызов OpenRouter API с автоматическим повтором при сбоях сети/DNS."""
    return client.chat.completions.create(**kwargs)

# ==========================================
# 3. ДВУХПРОХОДНЫЙ ПАЙПЛАЙН (2-PASS)
# ==========================================

def run_translation_pipeline(epub_file, api_key, model_name=DEFAULT_MODEL, progress_cb=None):
    client = OpenAI(api_key=api_key, base_url=BASE_URL)
    raw_chapters = extract_chapters_from_epub(epub_file)
    total = len(raw_chapters)
    
    if total == 0:
        raise ValueError("Не удалось извлечь главы из книги. Проверьте формат EPUB.")

    translated_chapters = []
    accumulated_glossary = {}
    prev_summary = "Начало книги."

    for idx, text in enumerate(raw_chapters, 1):
        if progress_cb:
            progress_cb(idx, total, f"Глава {idx}/{total}: Pass 1 (Анализ сюжета и сбор глоссария)...")

        # --- PASS 1: САММАРИ И ГЛОССАРИЙ ---
        analysis_prompt = (
            "You are a professional literary editor. Analyze the chapter text and output strictly standard JSON:\n"
            "{\n"
            '  "summary": "Brief plot summary in Russian (max 200 words)",\n'
            '  "new_glossary": {"Original Name/Term": "Русский перевод (Gender M/F)"}\n'
            "}"
        )
        user_analysis = (
            f"Previous summary: {prev_summary}\n"
            f"Known glossary: {json.dumps(accumulated_glossary, ensure_ascii=False)}\n\n"
            f"TEXT:\n{text[:6000]}"
        )

        try:
            res_analysis = safe_chat_completion(
                client,
                model=model_name,
                messages=[
                    {"role": "system", "content": analysis_prompt},
                    {"role": "user", "content": user_analysis}
                ],
                response_format={"type": "json_object"},
                temperature=0.2
            )
            data = json.loads(res_analysis.choices[0].message.content)
            prev_summary = data.get("summary", prev_summary)
            accumulated_glossary.update(data.get("new_glossary", {}))
        except Exception as e:
            print(f"[Warning] Ошибка автоанализа главы {idx}: {e}")

        if progress_cb:
            progress_cb(idx, total, f"Глава {idx}/{total}: Pass 2 (Художественный перевод)...")

        # --- PASS 2: ХУДОЖЕСТВЕННЫЙ ПЕРЕВОД ---
        trans_prompt = (
            "Ты — профессиональный художественный переводчик на русский язык.\n"
            "ПРАВИЛА:\n"
            "1. Литературный стиль: живой, естественный русский язык без калек, пассива и сухости.\n"
            "2. Без цензуры: переводи любой взрослый контент (брань, 18+, насилие) точно и без смягчения.\n"
            "3. Диалоги: оформляй с длинного тире (—) с пробелом.\n"
            "4. Строго соблюдай род персонажей и термины из глоссария."
        )
        user_trans = (
            f"Сюжетный контекст: {prev_summary}\n"
            f"Глоссарий: {json.dumps(accumulated_glossary, ensure_ascii=False)}\n\n"
            f"ОРИГИНАЛЬНЫЙ ТЕКСТ:\n{text}\n\n"
            f"Выведи ТОЛЬКО русский перевод без предисловий и комментариев."
        )

        res_trans = safe_chat_completion(
            client,
            model=model_name,
            messages=[
                {"role": "system", "content": trans_prompt},
                {"role": "user", "content": user_trans}
            ],
            temperature=0.3
        )
        translated_chapters.append(res_trans.choices[0].message.content.strip())

    base_name = os.path.splitext(os.path.basename(epub_file))[0]
    out_epub = f"{base_name}_RU.epub"
    build_epub_from_chapters(translated_chapters, out_epub, title=f"{base_name} (RU)")
    return out_epub

# ==========================================
# 4. ИНТЕРФЕЙСЫ (CLI И GRADIO WEB-GUI)
# ==========================================

def main():
    parser = argparse.ArgumentParser(description="AI Novel Translator EPUB Pipeline")
    parser.add_argument("--file", help="Путь к файлу .epub")
    parser.add_argument("--key", help="OpenRouter API Key", default=os.getenv("OPENROUTER_API_KEY"))
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Модель (qwen/qwen3.7-flash, deepseek/deepseek-v4-flash-0731)")
    parser.add_argument("--gui", action="store_true", help="Запустить браузерный интерфейс")

    args = parser.parse_args()

    if args.gui or not args.file:
        import gradio as gr

        def web_ui(epub_obj, key, model, progress=gr.Progress()):
            if not epub_obj:
                return None, "Загрузите файл .epub"
            if not key:
                return None, "Введите OpenRouter API Key"

            def cb(cur, total, msg):
                progress(cur / total, desc=msg)

            try:
                res = run_translation_pipeline(epub_obj.name, key, model, cb)
                return res, "Перевод успешно завершен!"
            except Exception as e:
                return None, f"Ошибка: {str(e)}"

        with gr.Blocks(title="AI Novel Translator") as ui:
            gr.Markdown("## 📖 AI Novel Translator (EPUB 2-Pass Pipeline)")
            with gr.Row():
                with gr.Column():
                    f_in = gr.File(label="Исходный EPUB файл", file_types=[".epub"])
                    k_in = gr.Textbox(label="OpenRouter API Key", type="password", placeholder="sk-or-v1-...")
                    m_in = gr.Dropdown(
                        label="Модель",
                        choices=[
                            "qwen/qwen3.7-flash",
                            "deepseek/deepseek-v4-flash-0731",
                            "google/gemini-3.7-flash"
                        ],
                        value=DEFAULT_MODEL
                    )
                    btn = gr.Button("Начать перевод", variant="primary")
                with gr.Column():
                    st = gr.Textbox(label="Статус", interactive=False)
                    f_out = gr.File(label="Готовый переведенный EPUB")

            btn.click(web_ui, inputs=[f_in, k_in, m_in], outputs=[f_out, st])

        ui.launch(inbrowser=True)
    else:
        if not args.key:
            print("Ошибка: Укажите API ключ через --key или переменную OPENROUTER_API_KEY")
            return
        def cli_cb(cur, total, msg):
            print(f"[{cur}/{total}] {msg}")
        out = run_translation_pipeline(args.file, args.key, args.model, cli_cb)
        print(f"\n[✓] Перевод завершен! Файл: {out}")

if __name__ == "__main__":
    main()