import streamlit as st
import tensorflow as tf
import cv2
import numpy as np
import tempfile
import os
from PIL import Image
import time
from pathlib import Path
from collections import deque
import librosa
import librosa.display
import matplotlib.pyplot as plt

# ------------------------------
# IMAGE + VIDEO DETECTOR CLASS
# ------------------------------
class DeepfakeDetector:
    def __init__(self, model_path):
        """Initialize the deepfake detector with a saved model."""
        self.model = tf.keras.models.load_model(model_path)
        self.img_height = 224
        self.img_width = 224
        self.window_size = 5
        self.confidence_threshold = 0.7

    def preprocess_image(self, image):
        """Preprocess image for model input with normalization."""
        if isinstance(image, str):
            img = tf.keras.preprocessing.image.load_img(
                image, target_size=(self.img_height, self.img_width)
            )
            img_array = tf.keras.preprocessing.image.img_to_array(img)
        else:
            img_array = cv2.resize(image, (self.img_height, self.img_width))
            if len(img_array.shape) == 2:
                img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)

        img_array = img_array / 255.0
        img_array = (img_array - np.mean(img_array)) / np.std(img_array)
        img_array = np.expand_dims(img_array, axis=0)
        return img_array

    def predict(self, image):
        """Make prediction with confidence score."""
        processed_img = self.preprocess_image(image)
        prediction = self.model.predict(processed_img, verbose=0)[0][0]
        confidence = abs(prediction - 0.5) * 2
        return prediction, confidence


# ------------------------------
# AUDIO DETECTOR CLASS
# ------------------------------
class AudioDeepfakeDetector:
    def __init__(self, model_path):
        """Initialize the audio deepfake detector with a saved model."""
        self.model = tf.keras.models.load_model(model_path)
        self.confidence_threshold = 0.7

    def create_mel_spectrogram(self, file_path):
        """Convert audio file to mel-scale spectrogram."""
        audio_data, sample_rate = librosa.load(file_path)
        mel_spectrogram = librosa.feature.melspectrogram(y=audio_data, sr=sample_rate)
        mel_decibel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)
        return mel_decibel_spectrogram

    def preprocess_audio(self, file_path):
        """Preprocess audio file for model input."""
        spectrogram = self.create_mel_spectrogram(file_path)
        spectrogram = np.expand_dims(spectrogram, axis=0)
        return spectrogram

    def predict(self, file_path):
        """Make prediction with confidence score for audio."""
        processed_audio = self.preprocess_audio(file_path)
        prediction = self.model.predict(processed_audio, verbose=0)[0][0]
        confidence = abs(prediction - 0.5) * 2
        return prediction, confidence


# ------------------------------
# FRAME EXTRACTION & ANALYSIS
# ------------------------------
def extract_frames(video_path, output_dir, frame_interval=2):
    """Extract frames with motion detection."""
    frames = []
    frame_paths = []
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    prev_frame = None
    motion_threshold = 30

    fps = cap.get(cv2.CAP_PROP_FPS)
    optimal_interval = max(1, int(fps / 4))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % optimal_interval == 0:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if prev_frame is not None:
                frame_delta = cv2.absdiff(cv2.cvtColor(prev_frame, cv2.COLOR_RGB2GRAY),
                                          cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY))
                motion_score = np.mean(frame_delta)
                if motion_score > motion_threshold:
                    frames.append(frame_rgb)
                    frame_path = os.path.join(output_dir, f'frame_{frame_count}.jpg')
                    cv2.imwrite(frame_path, cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))
                    frame_paths.append(frame_path)
            else:
                frames.append(frame_rgb)
                frame_path = os.path.join(output_dir, f'frame_{frame_count}.jpg')
                cv2.imwrite(frame_path, cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))
                frame_paths.append(frame_path)
            prev_frame = frame_rgb
        frame_count += 1
    cap.release()
    return frames, frame_paths, fps


def analyze_video_frames(detector, frames, progress_bar=None):
    """Analyze video frames with temporal smoothing."""
    results = []
    confidences = []
    prediction_window = deque(maxlen=detector.window_size)

    for i, frame in enumerate(frames):
        prediction, confidence = detector.predict(frame)
        prediction_window.append(prediction)
        smoothed_prediction = np.mean(prediction_window)
        results.append(smoothed_prediction)
        confidences.append(confidence)
        if progress_bar:
            progress_bar.progress((i + 1) / len(frames))
    return results, confidences


# ------------------------------
# MAIN STREAMLIT APP
# ------------------------------
def main():
    st.title("🔍 Enhanced Deepfake Detection System")

    # Model paths (update these as per your local setup)
    video_model_path = r"C:\Users\ruhet\Downloads\DeepFake Audio and Video-20251109T104526Z-1-001\DeepFake Audio and Video\deepfake-det\deepfake-det\Models\best_mobilenet_lstm_model.keras"
    audio_model_path = r"C:\Users\ruhet\Downloads\DeepFake Audio and Video-20251109T104526Z-1-001\DeepFake Audio and Video\deepfake-det\deepfake-det\Models\deepfake_audio_detector.h5"

    # Initialize detectors
    video_detector, audio_detector = None, None

    try:
        video_detector = DeepfakeDetector(video_model_path)
        st.success("✅ Video/Image model loaded successfully!")
    except Exception as e:
        st.error(f"Error loading image/video model: {str(e)}")

    try:
        audio_detector = AudioDeepfakeDetector(audio_model_path)
        st.success("✅ Audio model loaded successfully!")
    except Exception as e:
        st.error(f"Error loading audio model: {str(e)}")

    if video_detector is None and audio_detector is None:
        st.error("No models could be loaded. Please check your model paths.")
        return

    # ------------------------------
    # Tabs: Image | Audio | Video
    # ------------------------------
    tab1, tab2, tab3 = st.tabs(["🖼️ Image Detection", "🔉 Audio Detection", "🎥 Video Detection"])

    # ----------- IMAGE DETECTION -----------
    with tab1:
        st.header("🖼️ Image Deepfake Detection")

        uploaded_image = st.file_uploader(
            "Upload an image",
            type=['jpg', 'jpeg', 'png'],
            key="image_uploader"
        )

        if uploaded_image is not None and video_detector is not None:
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
                    tmp_file.write(uploaded_image.getvalue())
                    img_path = tmp_file.name

                img = Image.open(uploaded_image)
                st.image(img, caption="Uploaded Image", use_column_width=True)

                if st.button("Detect Deepfake", key="image_detect"):
                    with st.spinner("Analyzing image..."):
                        prediction, confidence = video_detector.predict(img_path)

                        st.write("### Results")
                        prob_fake = prediction * 100
                        prob_real = (1 - prediction) * 100

                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Probability of Fake", f"{prob_fake:.2f}%")
                        with col2:
                            st.metric("Probability of Real", f"{prob_real:.2f}%")
                        with col3:
                            st.metric("Confidence", f"{confidence * 100:.2f}%")

                        verdict = "FAKE" if prediction > 0.5 else "REAL"
                        if verdict == "FAKE":
                            st.error(f"### Final Verdict: {verdict}")
                        else:
                            st.success(f"### Final Verdict: {verdict}")

                        # if confidence < video_detector.confidence_threshold:
                        #     st.warning("⚠️ Low confidence prediction - result may not be reliable")

                os.unlink(img_path)
            except Exception as e:
                st.error(f"Error processing image: {str(e)}")

    # ----------- AUDIO DETECTION -----------
    with tab2:
        st.header("🔉 Audio Deepfake Detection")

        if audio_detector is None:
            st.error("Audio model not available.")
        else:
            uploaded_audio = st.file_uploader(
                "Upload an audio file",
                type=['wav', 'mp3', 'flac', 'm4a'],
                key="audio_uploader"
            )

            if uploaded_audio is not None:
                try:
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
                        tmp_file.write(uploaded_audio.getvalue())
                        audio_path = tmp_file.name

                    st.audio(uploaded_audio, format='audio/wav')

                    # Display waveform
                    try:
                        audio_data, sample_rate = librosa.load(audio_path)
                        fig, ax = plt.subplots(figsize=(12, 4))
                        librosa.display.waveshow(audio_data, sr=sample_rate, ax=ax)
                        ax.set_title("Audio Waveform")
                        st.pyplot(fig)
                        plt.close()

                        mel_spec = audio_detector.create_mel_spectrogram(audio_path)
                        fig, ax = plt.subplots(figsize=(12, 6))
                        librosa.display.specshow(mel_spec, x_axis='time', y_axis='mel', ax=ax)
                        plt.colorbar(ax.images[0], ax=ax, format='%+2.0f dB')
                        ax.set_title("Mel-Spectrogram")
                        st.pyplot(fig)
                        plt.close()
                    except Exception as viz_error:
                        st.warning(f"Could not visualize audio: {str(viz_error)}")

                    if st.button("Detect Deepfake", key="audio_detect"):
                        with st.spinner("Analyzing audio..."):
                            prediction, confidence = audio_detector.predict(audio_path)

                            st.write("### Results")
                            prob_fake = prediction * 100
                            prob_real = (1 - prediction) * 100
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Probability of Fake", f"{prob_fake:.2f}%")
                            with col2:
                                st.metric("Probability of Real", f"{prob_real:.2f}%")
                            with col3:
                                st.metric("Confidence", f"{confidence * 100:.2f}%")

                            verdict = "FAKE" if prediction > 0.5 else "REAL"
                            if verdict == "FAKE":
                                st.error(f"### Final Verdict: {verdict}")
                            else:
                                st.success(f"### Final Verdict: {verdict}")

                    os.unlink(audio_path)
                except Exception as e:
                    st.error(f"Error processing audio: {str(e)}")

    # ----------- VIDEO DETECTION -----------
    with tab3:
        st.header("🎥 Video Deepfake Detection")

        if video_detector is None:
            st.error("Video model not available.")
        else:
            uploaded_video = st.file_uploader(
                "Upload a video",
                type=['mp4', 'avi', 'mov'],
                key="video_uploader"
            )

            if uploaded_video is not None:
                try:
                    temp_dir = tempfile.mkdtemp()
                    video_path = os.path.join(temp_dir, "video.mp4")
                    with open(video_path, "wb") as f:
                        f.write(uploaded_video.read())

                    st.video(video_path)

                    if st.button("Detect Deepfake", key="video_detect"):
                        with st.spinner("Analyzing video..."):
                            frames, frame_paths, fps = extract_frames(video_path, temp_dir)
                            if not frames:
                                st.error("No frames could be extracted.")
                                return

                            progress_bar = st.progress(0)
                            results, confidences = analyze_video_frames(video_detector, frames, progress_bar)

                            weighted_prediction = np.average(results, weights=confidences)
                            avg_confidence = np.mean(confidences)
                            prob_fake = weighted_prediction * 100
                            prob_real = (1 - weighted_prediction) * 100

                            st.write("### Results")
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Weighted Probability of Fake", f"{prob_fake:.2f}%")
                            with col2:
                                st.metric("Weighted Probability of Real", f"{prob_real:.2f}%")
                            with col3:
                                st.metric("Average Confidence", f"{avg_confidence * 100:.2f}%")

                            st.write("### Frame Analysis Over Time")
                            plot_data = {
                                'Frame': list(range(len(results))),
                                'Fake Probability': [r * 100 for r in results],
                                'Confidence': [c * 100 for c in confidences]
                            }
                            st.line_chart(plot_data)

                            verdict = "FAKE" if weighted_prediction > 0.5 else "REAL"
                            if verdict == "FAKE":
                                st.error(f"### Final Verdict: {verdict}")
                            else:
                                st.success(f"### Final Verdict: {verdict}")

                            # Cleanup
                            for p in frame_paths:
                                os.unlink(p)
                            os.unlink(video_path)
                            os.rmdir(temp_dir)
                except Exception as e:
                    st.error(f"Error processing video: {str(e)}")


if __name__ == "__main__":
    main()
