import streamlit as st
import cv2

from main import run_surveillance

st.set_page_config(
    page_title="Multimodal Surveillance System",
    layout="wide",
)

st.title("Multimodal Surveillance System")
st.write("Monitor your surroundings using AI-powered surveillance.")

st.sidebar.title("Controls")
st.sidebar.write(
    "Start the surveillance system. The video and depth map will appear here."
)

video_column, depth_column = st.columns([2, 1])
with video_column:
    st.subheader("Live Surveillance")
    video_placeholder = st.empty()

with depth_column:
    st.subheader("Depth Map")
    depth_placeholder = st.empty()

status_placeholder = st.empty()


def display_in_streamlit(window_name, frame):
    """Show frames from the existing OpenCV code inside the Streamlit page."""
    if frame is None:
        return

    if window_name == "Depth Map":
        depth_placeholder.image(
            frame,
            channels="BGR",
            use_container_width=True,
        )
        return

    # Keep the displayed video reasonably sized without changing model input.
    height, width = frame.shape[:2]
    max_width = 960
    if width > max_width:
        new_height = int(height * max_width / width)
        frame = cv2.resize(frame, (max_width, new_height))

    video_placeholder.image(
        frame,
        channels="BGR",
        use_container_width=True,
    )


if st.button("Start Surveillance", type="primary"):
    status_placeholder.info("Surveillance is running. Keep this page open.")

    # main.py is left unchanged. Redirect its existing cv2.imshow calls to
    # Streamlit placeholders so the frames are shown in the browser.
    original_imshow = cv2.imshow
    original_wait_key = cv2.waitKey

    try:
        cv2.imshow = display_in_streamlit
        # No desktop OpenCV window is used in this interface.
        cv2.waitKey = lambda delay=0: -1
        run_surveillance()
    except Exception as e:
        status_placeholder.error(f"Something went wrong: {e}")
    finally:
        cv2.imshow = original_imshow
        cv2.waitKey = original_wait_key
        status_placeholder.info("Surveillance stopped.")

st.caption("Powered by YOLOv8, BLIP, and MiDaS")

