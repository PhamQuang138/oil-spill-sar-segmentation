import streamlit as st


st.set_page_config(
    page_title="Oil Spill SAR Segmentation",
    layout="wide",
)

st.title("Oil Spill SAR Segmentation")

st.sidebar.header("Inference")
uploaded_file = st.sidebar.file_uploader("Upload SAR image", type=["png", "jpg", "jpeg", "tif", "tiff"])
model_choice = st.sidebar.selectbox(
    "Model",
    ["Quang", "Hung", "Khoa", "Ensemble"],
)
threshold = st.sidebar.slider("Threshold", 0.0, 1.0, 0.5, 0.05)

if uploaded_file is None:
    st.info("Upload mot anh SAR de bat dau demo.")
else:
    st.warning("Inference se duoc tich hop sau khi 3 model va weight san sang.")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.subheader("Original")
        st.image(uploaded_file, use_container_width=True)
    with col2:
        st.subheader("Mask")
        st.empty()
    with col3:
        st.subheader("Overlay")
        st.empty()

