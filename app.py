import cv2
import streamlit as st

from main import run_surveillance


st.set_page_config(
    page_title="Multimodal Surveillance System",
    layout="wide",
)

st.title("Multimodal Surveillance System")
st.write(
    "Monitor your surroundings using AI-powered person detection, "
    "scene captions, depth estimation and voice alerts."
)

st.sidebar.title("Controls")
test_duration = st.sidebar.selectbox(
    "Test duration",
    options=[30, 60, 120],
    index=1,
    format_func=lambda seconds: f"{seconds} seconds",
)
st.sidebar.write(
    "The benchmark summary (average FPS and processing latency) is printed "
    "in the terminal running Streamlit."
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
    """Display existing OpenCV frames inside the Streamlit page."""
    if frame is None:
        return

    if window_name == "Depth Map":
        depth_placeholder.image(
            frame,
            channels="BGR",
            use_container_width=True,
        )
        return

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


if st.button("Start Surveillance Benchmark", type="primary"):
    status_placeholder.info(
        f"Surveillance is running for up to {test_duration} seconds."
    )

    # Route OpenCV's existing display calls to the Streamlit placeholders.
    original_imshow = cv2.imshow
    original_wait_key = cv2.waitKey

    try:
        cv2.imshow = display_in_streamlit
        # This browser interface has no OpenCV desktop keyboard window.
        cv2.waitKey = lambda delay=0: -1
        run_surveillance(max_seconds=test_duration)
        status_placeholder.success(
            "Benchmark finished. Check the terminal for measured FPS and latency."
        )
    except Exception as exc:
        status_placeholder.error(f"Something went wrong: {exc}")
    finally:
        cv2.imshow = original_imshow
        cv2.waitKey = original_wait_key

st.caption("Powered by YOLOv8, BLIP, and MiDaS.")

