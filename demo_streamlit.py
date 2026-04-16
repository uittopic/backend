#!/usr/bin/env python3
"""
Streamlit Demo cho BLIP Vietnamese Image Captioning

Demo thật sự - kết nối với API đang chạy.
Nếu API không chạy, sẽ hiển thị thông báo lỗi rõ ràng.
"""

import streamlit as st
import requests
import json
from pathlib import Path
from PIL import Image
import io
import time
import os

st.set_page_config(
    page_title="BLIP Vietnamese - Image Captioning",
    page_icon="🖼️",
    layout="centered"
)

# API Configuration
API_URL = os.environ.get("API_URL", "http://localhost:8000")
API_ENDPOINT = f"{API_URL}/api/caption_full"


def call_caption_api(image_bytes: bytes) -> dict:
    """
    Gọi API để tạo caption cho ảnh.

    Args:
        image_bytes: Bytes của ảnh

    Returns:
        dict với caption hoặc error
    """
    try:
        files = {"file": ("image.jpg", image_bytes, "image/jpeg")}
        response = requests.post(
            API_ENDPOINT,
            files=files,
            timeout=30
        )

        if response.status_code == 200:
            return {"success": True, "data": response.json()}
        else:
            return {
                "success": False,
                "error": f"API error: {response.status_code}",
                "detail": response.text
            }
    except requests.exceptions.ConnectionError:
        return {
            "success": False,
            "error": "Không thể kết nối API",
            "detail": f"Hãy đảm bảo API đang chạy tại {API_URL}"
        }
    except requests.exceptions.Timeout:
        return {
            "success": False,
            "error": "API timeout",
            "detail": "Yêu cầu mất quá lâu. Thử ảnh nhỏ hơn."
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "detail": "Lỗi không xác định"
        }


def check_api_health() -> bool:
    """Kiểm tra API có đang chạy không."""
    try:
        response = requests.get(f"{API_URL}/api/health", timeout=5)
        return response.status_code == 200
    except:
        return False


# UI
st.title("🖼️ BLIP Vietnamese Image Captioning")
st.markdown("---")

# Check API status
api_healthy = check_api_health()
if api_healthy:
    st.success("✅ API đang chạy - có thể tạo caption thật")
else:
    st.warning(f"⚠️ API không khả dụng tại {API_URL}. Demo sẽ hiển thị thông báo lỗi khi upload.")

st.write("""
Ứng dụng demo **Image Captioning tiếng Việt** được train trên dataset Shopee.
Model có thể tạo caption tự nhiên cho ảnh sản phẩm thương mại.

**Chế độ demo:**
- **Upload ảnh**: Gọi API thật để tạo caption
- **Xem kết quả test**: Hiển thị metrics và samples đã đánh giá
""")

mode = st.radio("Chọn chế độ:", ["📤 Upload ảnh", "📊 Xem kết quả test"])

if mode == "📤 Upload ảnh":
    st.header("Upload ảnh để lấy caption")

    uploaded_file = st.file_uploader(
        "Chọn ảnh...",
        type=['jpg', 'jpeg', 'png', 'webp']
    )

    if uploaded_file is not None:
        # Hiển thị ảnh
        image = Image.open(uploaded_file)
        st.image(image, caption="Ảnh đã upload", use_column_width=True)

        # Đọc bytes
        img_bytes = uploaded_file.getvalue()

        # Nút generate
        if st.button("🚀 Tạo caption", type="primary"):
            if not api_healthy:
                st.error("❌ API không khả dụng!")
                st.info(f"""
                **Vui lòng khởi động API trước:**

                ```bash
                cd /Users/nguyenhuuviet/UIT/Ky3/Chuyen_De/CHUYEN_DE_Backend
                uvicorn app.main:app --host 0.0.0.0 --port 8000
                ```

                Sau đó refresh trang này.
                """)
            else:
                with st.spinner("Đang xử lý..."):
                    start_time = time.time()
                    result = call_caption_api(img_bytes)

                    if result["success"]:
                        data = result["data"]
                        elapsed = time.time() - start_time

                        st.success(f"**Caption:** {data.get('caption_vi', 'N/A')}")

                        # Hiển thị thông tin thêm
                        col1, col2, col3 = st.columns(3)
                        col1.metric("Thời gian", f"{elapsed:.2f}s")
                        col2.metric("Device", data.get('device', 'N/A'))
                        col3.metric("Accent restored", "✅" if data.get('accent_restored') else "❌")

                        # Hiển thị caption không dấu nếu có
                        if data.get('caption_vi_no_accent'):
                            st.caption(f"Không dấu: {data['caption_vi_no_accent']}")
                    else:
                        st.error(f"❌ Lỗi: {result['error']}")
                        st.caption(f"Chi tiết: {result.get('detail', '')}")

elif mode == "📊 Xem kết quả test":
    st.header("Kết quả trên tập Test")

    # Load metrics
    metrics_path = Path(__file__).parent / "outputs" / "metrics.json"
    if metrics_path.exists():
        with open(metrics_path) as f:
            metrics = json.load(f)

        col1, col2, col3 = st.columns(3)
        col1.metric("BLEU-1", f"{metrics['bleu_scores']['average']['bleu1']*100:.1f}%")
        col2.metric("BLEU-4", f"{metrics['bleu_scores']['average']['bleu4']*100:.1f}%")
        col3.metric("Success Rate", metrics['batch_test']['success_rate'])

        st.markdown("---")

        # Quality distribution
        st.subheader("📈 Phân loại caption")
        quality = metrics['batch_test']['quality']
        total = metrics['batch_test']['total_images']

        data = {
            "Loại": ["✅ Good", "⚠️ Medium", "❌ Bad"],
            "Số lượng": [
                quality['good_captions'],
                quality['medium_captions'],
                quality['bad_captions']
            ],
            "Tỷ lệ": [
                f"{quality['good_captions']*100/total:.1f}%",
                f"{quality['medium_captions']*100/total:.1f}%",
                f"{quality['bad_captions']*100/total:.2f}%"
            ]
        }
        st.table(data)

        # Sample predictions
        st.subheader("📝 Một số mẫu dự đoán")
        samples = metrics.get('samples', [])[:10]

        for i, sample in enumerate(samples, 1):
            with st.expander(f"Mẫu #{i} - {sample['image'][:20]}..."):
                st.write(f"**Ground Truth:** {sample['ground_truth']}")
                st.write(f"**Prediction:** {sample['prediction']}")
                st.write(f"**BLEU-4:** {sample.get('bleu4', 0):.3f}")

    else:
        st.warning("Chưa có metrics file. Chạy `python tools/export_metrics.py` trước.")

st.markdown("---")
st.caption("BLIP Vietnamese - UIT Ky3 - 2026")

# Footer với hướng dẫn chạy API
with st.expander("📖 Hướng dẫn chạy API"):
    st.markdown("""
    **Bước 1: Khởi động API server**

    ```bash
    cd /Users/nguyenhuuviet/UIT/Ky3/Chuyen_De/CHUYEN_DE_Backend

    # Option 1: Chạy trực tiếp
    uvicorn app.main:app --host 0.0.0.0 --port 8000

    # Option 2: Chạy với reload (dev)
    uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
    ```

    **Bước 2: Chạy Streamlit demo**

    ```bash
    # Terminal khác
    streamlit run demo_streamlit.py
    ```

    **Bước 3: Mở trình duyệt**

    Truy cập: http://localhost:8501
    """)
