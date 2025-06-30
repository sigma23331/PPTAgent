import os
import json
from pathlib import Path
from openai import OpenAI

client = OpenAI(api_key="sk-9dacab67e4a34a749edb24357e4afbe0", base_url="https://api.deepseek.com")

def get_template_by_category(page_category, template_dir):
    mapping = {
        "标题页": "slide_1.html",
        "目录页": "slide_2.html",
        "章节标题页": "slide_3.html",
        "C": "slide_4.html",
        "B": "slide_5.html",
        "A": "slide_6.html",
        "结尾页": "slide_7.html",
        # 你可以继续扩展其他类型
    }
    template_file = mapping.get(page_category, "slide_content.html")
    template_path = Path(template_dir) / template_file
    if template_path.exists():
        with open(template_path, "r", encoding="utf-8") as f:
            return f.read()
    else:
        return ""  # 或者抛出异常
    
def ask_ai_page_type(page):
    # 让AI判断应该用哪种页面类型
    prompt = (
        f"{json.dumps(page, ensure_ascii=False, indent=2)}"
    )
    with open("select.txt", "r", encoding="utf-8") as f:
        select_prompt = f.read()
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": select_prompt},
            {"role": "user", "content": prompt}
        ],
        response_format={"type": "text"},
        stream=False
    )
    # 只取AI返回的类型名称
    return response.choices[0].message.content.strip()

def extract_pages_content(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    pages = data.get("pages", [])
    result = []
    for page in pages:
        page_info = {
            "pageCategory": page.get("pageCategory"),
            "specialContent": page.get("specialContent"),
            "pageTheme": page.get("pageTheme"),
            "chapterNumber": page.get("chapterNumber"),
            "content": page.get("content"),
            "figure": page.get("figure"),
        }
        result.append(page_info)
    return result

def read_html_slides(slide_dir):
    slides = []
    slide_files = sorted(Path(slide_dir).glob("slide_*.html"), key=lambda x: int(x.stem.split('_')[1]))
    for slide_file in slide_files:
        with open(slide_file, "r", encoding="utf-8") as f:
            slides.append(f.read())
    return slides, slide_files

def generate_page(system_prompt, user_prompt):
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        response_format={"type": "text"},
        stream=False
    )
    return response.choices[0].message.content

def load_prompt_mapping(prompt_json_path):
    with open(prompt_json_path, "r", encoding="utf-8") as f:
        prompt_list = json.load(f)
    # 构建类型到prompt的映射字典
    return {item["type"]: item["prompt"] for item in prompt_list}

def main():
    template_dir = "html_slides"
    page_content_path = "ppt_outline_updated.json"
    prompt_json_path = "prompt.json"
    output_dir = "generated_ppt"
    os.makedirs(output_dir, exist_ok=True)

    # 加载类型到prompt的映射
    prompt_mapping = load_prompt_mapping(prompt_json_path)
    pages = extract_pages_content(page_content_path)

    for idx, page in enumerate(pages):
        page_category = page.get("pageCategory")
        # 针对具体内容页，先让AI判断类型
        if page_category == "具体内容页":
            ai_type = ask_ai_page_type(page)
            template_type = ai_type
        else:
            template_type = page_category
        slide_html = get_template_by_category(template_type, template_dir)
        # 获取对应的prompt（如果是结尾页，则不用使用AI，直接输出HTML）
        if template_type == "结尾页":
            html_result = slide_html
        # 如果是章节标题页，则要在promt后追加第几章
        elif template_type == "章节标题页":
            chapter_number =  page.get("chapterNumber")
            user_prompt = prompt_mapping.get(template_type, "请将大纲内容填充到模板中，生成完整HTML页面。")
            system_prompt = (
                f"【HTML模板如下】\n{slide_html}\n\n"
                f"【页面大纲如下】\n{json.dumps(page, ensure_ascii=False, indent=2)}\n\n"
                f"【章节编号】\n{chapter_number}"
            )
            html_result = generate_page(system_prompt, user_prompt)
        else:
            user_prompt = prompt_mapping.get(template_type, "请将大纲内容填充到模板中，生成完整HTML页面。")
            system_prompt = (
                f"【HTML模板如下】\n{slide_html}\n\n"
                f"【页面大纲如下】\n{json.dumps(page, ensure_ascii=False, indent=2)}"
            )
            html_result = generate_page(system_prompt, user_prompt)
        out_file = Path(output_dir) / f"page_{idx+1}.html"
        with open(out_file, "w", encoding="utf-8") as f:
            f.write(html_result)
        print(f"已生成：{out_file}")

if __name__ == "__main__":
    main()