
import streamlit as st
from main import run_surveillance

st.set_page_config(
    page_title="Multimodal Surveillance System",
    layout="wide"
)

st.title("Multimodal Surveillance System")
st.write("Monitor your surroundings using AI-powered surveillance.")

st.sidebar.title("Controls")
st.sidebar.write("Start the surveillance system when you're ready.")

if st.button("Start Surveillance", type="primary"):
    with st.spinner("Starting surveillance..."):
        try:
            run_surveillance()
        except Exception as e:
            st.error(f"Something went wrong: {e}")

st.caption("Powered by YOLOv8, BLIP, and MiDaS")


