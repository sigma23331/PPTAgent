import os
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from PIL import Image
import time


# 设置浏览器驱动（以 Chrome 为例）
driver_path = './chromedriver.exe'  # 替换为你的 ChromeDriver 路径
service = Service(driver_path)
service.start()

# 设置无头模式
options = Options()
options.headless = True  # 启用无头模式，浏览器不显示界面
options.add_argument('--disable-gpu')  # 禁用 GPU 加速（可选，但有时可以解决一些问题）
options.add_argument('--no-sandbox')  # 防止某些操作系统的安全机制干扰
options.add_argument('window-size=1280x1024')  # 设置窗口大小（可选）
options.add_argument('start-maximized')  # 启动时最大化窗口

# 确保无头模式下完全运行
options.add_argument('disable-software-rasterizer')  # 避免一些渲染问题

# 启动无头浏览器
driver = webdriver.Chrome(service=service, options=options)

# 确保 result 文件夹存在
if not os.path.exists('result'):
    os.makedirs('result')

# 目标文件夹路径
input_folder = './generated_ppt'

# 遍历文件夹中的所有 .html 文件
for file_name in os.listdir(input_folder):
    if file_name.endswith('.html'):  # 只处理 .html 文件
        try:
            # 构建文件路径
            file_path = 'file://' + os.path.abspath(os.path.join(input_folder, file_name))
            driver.get(file_path)

            # 等待页面加载完成
            time.sleep(2)  # 可以根据需要增加等待时间

            # 调整窗口大小为整个页面大小
            driver.set_window_size(1294, 866)

            # 截图操作并保存到 result 文件夹
            screenshot_path = os.path.join('result', f'{os.path.splitext(file_name)[0]}.png')
            driver.save_screenshot(screenshot_path)
            print(f"截图已保存: {screenshot_path}")

        except Exception as e:
            print(f"处理文件 {file_name} 时发生错误: {e}")

# 关闭浏览器
driver.quit()