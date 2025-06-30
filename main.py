import shutil
import os

# 运行pdf_to_json.py
os.system('python pdf_to_json.py')

# 获取uploads文件夹中pdf文件的文件名
def get_pdf_filename(upload_folder):
    for filename in os.listdir(upload_folder):
        if filename.endswith('.pdf'):
            return filename
    return None

upload_folder = 'uploads'
pdf_filename = get_pdf_filename(upload_folder)
# 去掉pdf后缀
pdf_filename = pdf_filename[:-4] if pdf_filename else None
# 检查是否找到了PDF文件
if pdf_filename is None:
    print("未找到PDF文件，请确保uploads文件夹中有PDF文件。")
    exit(1)

# 定义源文件夹和目标文件夹路径
image_source_folder = 'output\\' + pdf_filename + '\\auto\\images'
image_destination_folder = 'static\\images'
image_another_destination_folder ='generated_ppt\\images'

# 如果目标文件夹已经存在，则删除它
if os.path.exists(image_destination_folder):
    shutil.rmtree(image_destination_folder)

# 如果目标文件夹已经存在，则删除它
if os.path.exists(image_another_destination_folder):
    shutil.rmtree(image_another_destination_folder)

# 将源文件夹复制一份放到image_another_destination_folder中
shutil.copytree(image_source_folder, image_another_destination_folder)

# 移动文件夹
shutil.move(image_source_folder, image_destination_folder)

print(f"'{image_source_folder}' 已成功移动到 '{image_destination_folder}' 并覆盖同名文件夹。")

#移动文件
content_source_file = 'output\\' + pdf_filename + '\\auto\\' + pdf_filename + '_content_list.json'
content_destination_file = 'paper_content_list.json'

# 如果目标文件已经存在，则删除它
if os.path.exists(content_destination_file):
    os.remove(content_destination_file)

# 移动文件
shutil.move(content_source_file, content_destination_file)

print(f"'{content_source_file}' 已成功移动到 '{content_destination_file}' 并覆盖同名文件。")

# 将这个文件更名为'paper_content_list.json'
os.rename(content_destination_file, 'paper_content_list.json')

# 运行outline_generator.py
os.system('python outline_generator.py')

# 如果generated_ppt目录中已经存在html文件，则删除全部html文件
html_dir = 'generated_ppt'
if os.path.exists(html_dir):
    for file in os.listdir(html_dir):
        if file.endswith('.html'):
            os.remove(os.path.join(html_dir, file))

# 运行build_single_slide.py
os.system('python build_single_slide.py')




