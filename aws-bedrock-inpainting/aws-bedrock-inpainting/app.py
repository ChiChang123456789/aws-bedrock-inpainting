
import streamlit as st
import boto3
import json
import base64
from io import BytesIO

st.set_page_config(page_title="AWS Bedrock Inpainting", layout="wide")
st.title("🖌️ AWS Bedrock - Inpainting (Object Removal)")

# 建立 AWS Bedrock 客戶端
session = boto3.Session(
    aws_access_key_id=st.secrets["aws"]["aws_access_key_id"],
    aws_secret_access_key=st.secrets["aws"]["aws_secret_access_key"],
    region_name=st.secrets["aws"]["region_name"]
)
client = session.client('bedrock-runtime')

# 上傳圖片
uploaded_file = st.file_uploader("上傳圖片", type=["jpg", "jpeg", "png"])
if uploaded_file:
    image_bytes = uploaded_file.read()
    st.image(image_bytes, caption="原始圖片", use_column_width=True)

    # 使用者輸入 mask 描述
    prompt = st.text_input("輸入要移除/取代的物件描述（英文）", "remove the person")

    if st.button("執行移除"):
        # Base64 編碼圖片
        encoded_image = base64.b64encode(image_bytes).decode("utf-8")

        # 設定輸入資料格式
        request = {
            "taskType": "INPAINT",
            "image": encoded_image,
            "maskPrompt": prompt
        }

        # 呼叫 Bedrock 模型（範例以 Titan Image Inpainting 為例）
        response = client.invoke_model(
            modelId="amazon.titan-image-generator-v1",
            body=json.dumps(request),
            contentType="application/json",
            accept="application/json"
        )

        result = json.loads(response['body'].read())
        output_image = base64.b64decode(result["images"][0])

        st.image(output_image, caption="處理後圖片", use_column_width=True)
