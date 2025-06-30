import os
from pathlib import Path
from flask import Flask, render_template_string, request, redirect, url_for
from bs4 import BeautifulSoup
from flask import send_file, flash
from werkzeug.utils import secure_filename
import subprocess  # Import the subprocess module
import shutil
from flask import request, jsonify
import base64
import tempfile
from flask import make_response  # 添加这行导入
from PIL import Image
# -*- coding: utf-8 -*-
import io
from flask import send_from_directory
import time

UPLOAD_FOLDER = "uploads"
ALLOWED_EXTENSIONS = {"pdf"}
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = Flask(__name__)
PPT_DIR = "E:/PPTAgent/generated_ppt/"

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.secret_key = "some_secret"  # Required for flash messages

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def load_all_html_pages(ppt_dir):
    html_files = sorted(Path(ppt_dir).glob("page_*.html"), key=lambda x: int(x.stem.split('_')[1]))
    pages = []
    for file in html_files:
        with open(file, "r", encoding="utf-8") as f:
            content = f.read()
            content = content.replace('src="images/', 'src="/static/images/')
            content = content.replace("src='images/", "src='/static/images/")
            pages.append(content)
    return pages, html_files

def extract_text_nodes(html):
    soup = BeautifulSoup(html, 'html.parser')
    text_nodes = []
    for element in soup.find_all(string=True):
        if element.parent.name not in ['script', 'style'] and element.strip():
            text_nodes.append(element)
    return soup, text_nodes

@app.route("/upload", methods=["GET", "POST"])
def upload_pdf():
    message = ""
    view_button = ""
    uploaded_filename = ""

    if request.method == "POST":
        if "pdf_file" not in request.files:
            message = "未找到文件字段。"
        else:
            file = request.files["pdf_file"]
            if file.filename == "":
                message = "未选择文件。"
            elif file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                save_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
                file.save(save_path)

                uploaded_filename = filename
                message = f"成功上传文件：{filename}"

                # Call main.py after successful upload
                flash("正在生成PPT，请稍候...", "info")  # Display "Please wait" message
                return redirect(url_for("process_pdf"))  # Redirect to processing route
            else:
                message = "仅支持 PDF 文件上传。"

    html = f"""
    <html>
    <head>
        <meta charset="utf-8">
        <title>上传PDF文件</title>
        <style>
            body {{
                font-family: 'Segoe UI', 'Roboto', 'Arial', sans-serif;
                background: linear-gradient(135deg, #141e30 0%, #243b55 60%, #6a3093 100%);
                min-height: 100vh;
                margin: 0;
                overflow: hidden;
                color: #eaf6ff;
            }}
            .container {{
                border: 1px solid rgba(106,48,147,0.18);
                padding: 36px 32px;
                max-width: 600px;
                margin: 60px auto;
                border-radius: 12px;
                background: rgba(30, 32, 40, 0.92);
                color: #eaf6ff;
                box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
                backdrop-filter: blur(8px);
            }}
            h2 {{
                color: #eaf6ff;
                text-shadow: 0 0 12px #8f6fff, 0 0 2px #fff;
                margin-bottom: 24px;
            }}
            input[type="file"] {{
                margin-bottom: 15px;
                color: #eaf6ff;
            }}
            button {{
                padding: 10px 28px;
                background: linear-gradient(90deg, #8f6fff 0%, #005bea 100%);
                color: #eaf6ff;
                border: none;
                border-radius: 8px;
                font-size: 16px;
                font-weight: bold;
                box-shadow: 0 0 16px #8f6fff, 0 2px 8px rgba(0,0,0,0.2);
                cursor: pointer;
                transition: background 0.3s, color 0.3s, box-shadow 0.3s;
            }}
            button:hover {{
                background: linear-gradient(90deg, #005bea 0%, #8f6fff 100%);
                color: #fff;
                box-shadow: 0 0 24px #8f6fff, 0 4px 16px rgba(0,0,0,0.3);
            }}
            a {{
                color: #8f6fff;
                text-decoration: underline;
            }}
            p {{
                color: #eaf6ff;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <h2>上传PDF文件</h2>
            <form method="post" enctype="multipart/form-data">
                <input type="file" name="pdf_file" accept=".pdf" required><br>
                <button type="submit">上传</button>
            </form>
            <p style="color: #00ff99;">{message}</p>
            {view_button}
            <div style="margin-top:18px;">
                <a href="/">← 返回主页面</a>
            </div>
        </div>
    </body>
    </html>
    """
    return html

@app.route("/process_pdf")
def process_pdf():
    # Execute main.py
    try:
        # Assuming main.py is in the same directory as merge_slides.py
        subprocess.run(["python", "main.py"], check=True, cwd=os.path.dirname(os.path.abspath(__file__)))
        flash("PDF处理完成！", "success")
        return redirect(url_for("index"))  # Redirect to editing page
    except subprocess.CalledProcessError as e:
        flash(f"main.py 执行失败：{e}", "error")
        return redirect(url_for("home"))
    except FileNotFoundError:
        flash("main.py 文件未找到，请确保它与 merge_slides.py 在同一目录下。", "error")
        return redirect(url_for("home"))

@app.route("/edit", methods=["GET", "POST"])
def index():
    pages, html_files = load_all_html_pages(PPT_DIR)

    export_status = request.args.get("export_status", "")
    export_ready = request.args.get("export_ready", "")

    if request.method == "POST":
        idx = int(request.form["edit_idx"])
        page_html = pages[idx]
        soup, text_nodes = extract_text_nodes(page_html)

        for i, node in enumerate(text_nodes):
            new_text = request.form.get(f"text_{i}")
            if new_text is not None:
                node.replace_with(new_text)

        new_html = str(soup)
        with open(html_files[idx], "w", encoding="utf-8") as f:
            f.write(new_html)
        return redirect(url_for("index"))

    merged_html = """
    <html>
    <head>
        <meta charset="utf-8">
        <title>合并PPT页面 - 文本编辑模式</title>
        <style>
            body { font-family: sans-serif; padding: 20px; }
            .ppt-page { border: 1px solid #ccc; margin: 60px 0; padding: 30px; background: #f9f9f9; border-radius: 8px; }
            iframe { width: 1280px; height: 720px; border: 1px solid #999; margin-bottom: 10px; background: #fff; }
            textarea { width: 1560px; min-height: 60px; margin-bottom: 10px; }
            .edit-form { display: none; margin-top: 10px; }
            .edit-btn { padding: 6px 12px; }
            .export-btn { padding: 8px 18px; background: #28a745; color: #fff; border: none; border-radius: 5px; margin-right: 10px; }
            .export-btn:disabled { background: #aaa; }
        </style>
        <script>
        function toggleEditForm(idx) {
            const form = document.getElementById("form_" + idx);
            if (form.style.display === "none") {
                form.style.display = "block";
            } else {
                form.style.display = "none";
            }
        }
        </script>
    </head>
    <body>
        <h1>生成的PPT（可修改）</h1>
        <div class="export-btns">
            <form method="post" action="/export_ppt" style="display:inline;">
                <button type="submit" class="export-btn">导出为PPT</button>
            </form>
    """

    if export_status == "processing":
        merged_html += "<span style='color:orange;'>正在导出，请稍候...</span>"
    if export_ready == "1":
        merged_html += """
        <form method="get" action="/download_pptx" style="display:inline;">
            <button type="submit" class="export-btn" style="background:#007bff;">导出已完成，点击下载</button>
        </form>
        """

    merged_html += "</div>"

    for idx, html in enumerate(pages):
        soup, text_nodes = extract_text_nodes(html)
        merged_html += f"""
        <div class="ppt-page">
            <h2>第{idx+1}页</h2>
            <iframe src="/view_page/{idx}"></iframe>
            <button class="edit-btn" onclick="toggleEditForm({idx})">修改文本</button>
            <form method="post" class="edit-form" id="form_{idx}">
                <input type="hidden" name="edit_idx" value="{idx}">
        """

        for i, node in enumerate(text_nodes):
            merged_html += f"<label>段落 {i+1}：</label><textarea name='text_{i}'>{node.strip()}</textarea>"

        merged_html += """
                <button type="submit">保存修改</button>
            </form>
        </div>
        """

    merged_html += "</body></html>"
    return render_template_string(merged_html)

@app.route("/view_page/<int:idx>")
def view_page(idx):
    _, html_files = load_all_html_pages(PPT_DIR)
    if 0 <= idx < len(html_files):
        with open(html_files[idx], "r", encoding="utf-8") as f:
            content = f.read()
            content = content.replace('src="images/', 'src="/static/images/')
            content = content.replace("src='images/", "src='/static/images/")
        return content
    return "页面不存在", 404

@app.route("/")
def home():
    # Clear uploads directory
    folder = app.config['UPLOAD_FOLDER']
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))

    # Display home page
    html = """
    <html>
    <head>
        <meta charset="utf-8">
        <title>PPTAgent——你的学术汇报助手</title>
        <style>
            body {
                font-family: 'Segoe UI', 'Roboto', 'Arial', sans-serif;
                background: linear-gradient(135deg, #141e30 0%, #243b55 60%, #6a3093 100%);
                min-height: 100vh;
                margin: 0;
                overflow: hidden;   /* 确保没有滚动条 */
                position: relative;
                color: #eaf6ff; /* 全局浅色字体 */
            }
            .container {
                text-align: center;
                padding: 48px 64px;
                border-radius: 16px;
                backdrop-filter: blur(8px);
                border: 1px solid rgba(106,48,147,0.18);
                position: relative;
                z-index: 2;
                color: #eaf6ff; /* 容器内浅色字体 */
            }
            h1 {
                margin-bottom: 28px;
                color: #eaf6ff; /* 更浅色 */
                letter-spacing: 2px;
                text-shadow: 0 0 18px #8f6fff, 0 0 2px #fff;
                font-size: 2.5em;
                font-weight: bold;
            }
            .btn {
                display: inline-block;
                padding: 14px 32px;
                margin: 12px 16px 0 16px;
                background: linear-gradient(90deg, #8f6fff 0%, #005bea 100%);
                color: #eaf6ff; /* 按钮文字更浅 */
                text-decoration: none;
                border-radius: 8px;
                font-size: 18px;
                font-weight: bold;
                box-shadow: 0 0 24px #8f6fff, 0 2px 8px rgba(0,0,0,0.2);
                border: none;
                transition: background 0.3s, color 0.3s, box-shadow 0.3s, transform 0.2s;
                cursor: pointer;
            }
            .btn:hover {
                background: linear-gradient(90deg, #005bea 0%, #8f6fff 100%);
                color: #fff;
                box-shadow: 0 0 32px #8f6fff, 0 4px 16px rgba(0,0,0,0.3);
                transform: translateY(-2px) scale(1.04);
            }
            .flash {
                padding: 12px;
                margin-bottom: 18px;
                border-radius: 6px;
                font-size: 16px;
            }
            .flash.info {
                background-color: #232526;
                color: #8f6fff;
                border: 1px solid #8f6fff;
                box-shadow: 0 0 8px #8f6fff;
            }
            .flash.success {
                background-color: #232526;
                color: #00ff99;
                border: 1px solid #00ff99;
                box-shadow: 0 0 8px #00ff99;
            }
            .flash.error {
                background-color: #232526;
                color: #ff4b2b;
                border: 1px solid #ff4b2b;
                box-shadow: 0 0 8px #ff4b2b;
            }
            #particles-bg {
                position: fixed;
                top: 0; left: 0; right: 0; bottom: 0;
                width: 100vw;
                height: 100vh;
                z-index: 0;
                pointer-events: none;
            }
        </style>
    </head>
    <body>
        <canvas id="particles-bg"></canvas>
        <div class="container">
            <h1>学术风格PPT生成</h1>
            {% with messages = get_flashed_messages(with_categories=true) %}
              {% if messages %}
                {% for category, message in messages %}
                  <div class="flash {{ category }}">{{ message }}</div>
                {% endfor %}
              {% endif %}
            {% endwith %}
            <a class="btn" href="/upload">上传 PDF 文件</a>
            <a class="btn" href="/edit">前往PPT编辑页面</a>
        </div>
        <script>
        // 深蓝+紫色粒子动效
        const canvas = document.getElementById('particles-bg');
        const ctx = canvas.getContext('2d');
        let particles = [];
        let ripples = []; // 存储波纹

        function resize() {
            canvas.width = window.innerWidth;
            canvas.height = window.innerHeight;
        }
        window.addEventListener('resize', resize);
        resize();

        // 粒子颜色为蓝紫渐变
        const colors = [
            'rgba(143,111,255,0.8)', // 紫
            'rgba(0,91,234,0.7)',    // 蓝
            'rgba(106,48,147,0.7)',  // 深紫
            'rgba(36,59,85,0.7)'     // 深蓝
        ];
        for(let i=0;i<80;i++){
            particles.push({
                x: Math.random()*canvas.width,
                y: Math.random()*canvas.height,
                r: Math.random()*2+1,
                dx: (Math.random()-0.5)*0.7,
                dy: (Math.random()-0.5)*0.7,
                color: colors[Math.floor(Math.random()*colors.length)]
            });
        }

        // 波纹动画
        canvas.addEventListener('mousedown', function(e){
            const rect = canvas.getBoundingClientRect();
            const x = e.clientX - rect.left;
            const y = e.clientY - rect.top;
            ripples.push({
                x: x,
                y: y,
                radius: 0,
                alpha: 0.5,
                max: 120
            });
        });

        function draw(){
            ctx.clearRect(0,0,canvas.width,canvas.height);
            // 粒子
            for(let p of particles){
                ctx.beginPath();
                ctx.arc(p.x,p.y,p.r,0,2*Math.PI);
                ctx.fillStyle=p.color;
                ctx.shadowColor=p.color;
                ctx.shadowBlur=12;
                ctx.fill();
                p.x+=p.dx;
                p.y+=p.dy;
                if(p.x<0||p.x>canvas.width) p.dx*=-1;
                if(p.y<0||p.y>canvas.height) p.dy*=-1;
            }
            // 波纹
            for(let i=0;i<ripples.length;i++){
                let ripple = ripples[i];
                ctx.save();
                let grad = ctx.createRadialGradient(ripple.x, ripple.y, ripple.radius*0.2, ripple.x, ripple.y, ripple.radius);
                grad.addColorStop(0, `rgba(0,234,255,${ripple.alpha})`);
                grad.addColorStop(0.5, `rgba(143,111,255,${ripple.alpha*0.5})`);
                grad.addColorStop(1, 'rgba(0,0,0,0)');
                ctx.beginPath();
                ctx.arc(ripple.x, ripple.y, ripple.radius, 0, 2*Math.PI);
                ctx.fillStyle = grad;
                ctx.fill();
                ctx.restore();
                ripple.radius += 3;
                ripple.alpha *= 0.96;
                if(ripple.radius > ripple.max || ripple.alpha < 0.05){
                    ripples.splice(i,1);
                    i--;
                }
            }
            requestAnimationFrame(draw);
        }
        draw();
        </script>
    </body>
    </html>
    """
    return render_template_string(html)


@app.route("/export_ppt", methods=["POST"])
def export_ppt():
    # 执行 get_picture.py
    subprocess.run(["python", "get_picture.py"], check=True, cwd=os.path.dirname(os.path.abspath(__file__)))
    # 执行 ceate_pptx.py
    subprocess.run(["python", "create_pptx.py"], check=True, cwd=os.path.dirname(os.path.abspath(__file__)))
    # 等待 results/combined_presentation.pptx 文件生成
    pptx_path = os.path.join("result", "combined_presentation.pptx")
    for _ in range(60):  # 最多等60秒
        if os.path.exists(pptx_path):
            break
        time.sleep(1)
    else:
        return redirect(url_for("index", export_status="fail"))
    return redirect(url_for("index", export_ready="1"))

@app.route("/download_pptx")
def download_pptx():
    pptx_path = os.path.join("result", "combined_presentation.pptx")
    if os.path.exists(pptx_path):
        return send_from_directory("result", "combined_presentation.pptx", as_attachment=True)
    return "PPTX文件不存在", 404

if __name__ == "__main__":
    app.run(debug=True)
