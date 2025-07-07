import streamlit as st
from PIL import Image, ImageDraw, ImageFont
import io
from vit_mask_pre1 import getModel, GetBoxesByPath

# 这里是你自己的模型生成函数，增加 language 参数
def generate_meme_text(image, language):
    """
    假设这是你自己的模型函数，输入图片和语言，输出生成的meme文本。
    这里只是示例，根据语言返回不同文本。
    """
    if language == "中文":
        return "这个meme有点意思！"
    else:
        return "This meme is kinda funny!"

# 设置页面标题
st.title("Meme生成器🎉")
st.write("上传无字图片，为你配字！")

# 语言选择
language = st.selectbox("请选择生成语言", ["中文", "English"])

# 文件上传器
uploaded_file = st.file_uploader("选择一张图片", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # 读取上传的图片
    image = Image.open(uploaded_file)

    if language == "中文":
        # 加载模型
        model = getModel(language='ch')
        # 得到boxe，每个文本框为 (x1, y1, x2, y2) 坐标
        boxes = GetBoxesByPath(image, model)
    else:
        # 加载模型
        model = getModel(language='en')
        # 得到boxe，每个文本框为 (x1, y1, x2, y2) 坐标
        boxes = GetBoxesByPath(image, model)

    # 展示用户上传的图片
    st.image(image, caption="你上传的图片", use_column_width=True)

    # 生成meme文本
    with st.spinner("正在努力配字中..."):
        meme_text = generate_meme_text(image, language)

    # 在页面中展示生成的meme文本
    st.markdown(f"**生成的meme文本：** {meme_text}")

    # 复制图片，避免修改原图
    meme_image = image.copy()
    draw = ImageDraw.Draw(meme_image)
    
    # 设置字体（路径视情况修改）
    try:
        font = ImageFont.truetype("arial.ttf", size=40)
    except:
        font = ImageFont.load_default()

    # 在图片上绘制文本
    text_position = (10, 10)
    text_color = (255, 255, 255)  # 白色
    draw.text(text_position, meme_text, font=font, fill=text_color, stroke_width=2, stroke_fill=(0, 0, 0))

    # 显示最终meme图片
    st.image(meme_image, caption="生成的Meme图", use_column_width=True)

    # 提供下载链接
    buf = io.BytesIO()
    meme_image.save(buf, format="PNG")
    byte_im = buf.getvalue()
    st.download_button(label="下载生成的Meme图片",
                       data=byte_im,
                       file_name="generated_meme.png",
                       mime="image/png")
