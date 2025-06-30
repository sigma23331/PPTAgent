# PPTAgent——学术PPT生成助手
## 项目简介
基于文本图像识别技术与大语言模型的学术风格PPT自动生成工具，希望给科研工作者阅读文献、汇报文献提供帮助
## 功能介绍
* 给定一个PDF格式的学术论文，一键生成学术风格的PPT，无需繁琐的提示词
* 支持修改PPT的文字内容
* 支持一键导出为.pptx文件
## 使用方法
* 安装相关环境(先requirements.txt);注意chromedriver.exe是否与当前谷歌版本对应
* 由于需要调用大模型API，请先将build_single_slide和poutline_generator中的API换成自己的
* 运行`merge_slides.py`文件，生成一个网址链接，进入网址链接
* 在“上传PDF文件”页面中上传要处理的论文文件
* 等待PPT的构建（可关注终端以了解处理状态）
* 完成后进入PPT编辑页面，如果有想编辑的内容直接点击编辑页面按钮，在弹出的文本框中进行文本修改，修改后点击“保存修改”。
* 若修改完毕，直接点击“导出为PPT”按钮，即可下载完整的.pptx格式文件
## Acknowledgement
- [MinerU](https://github.com/opendatalab/MinerU)
