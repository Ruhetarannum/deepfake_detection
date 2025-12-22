# DeFake AI: A Multimodal Deepfake Detection System

**An AI-powered web application for detecting deepfake content across images, videos, and audio using deep learning techniques.**

---

## 📋 Project Overview

DeFake AI is a comprehensive deepfake detection system designed to identify manipulated media content across multiple modalities. The system leverages advanced deep learning architectures to analyze and classify images, videos, and audio files as either genuine or deepfake. Built as a final-year engineering project, this application addresses the growing concern of synthetic media manipulation in the digital age.

The project implements a user-friendly web interface using Streamlit, allowing users to upload media files and receive instant detection results with confidence scores.

---

## 🎯 Problem Statement

The rapid advancement of generative AI technologies has made it increasingly easy to create realistic deepfake content, leading to serious concerns around misinformation, identity theft, fraud, and security threats. Traditional detection methods struggle to keep pace with evolving deepfake generation techniques.

There is a critical need for robust, accessible, and multimodal detection systems that can identify manipulated content across different media types with high accuracy.

---

## 💡 Proposed Solution

DeFake AI offers a unified platform for deepfake detection across three major media types:

- **Image Detection**: Identifies face-swapped or AI-generated images using a hybrid MobileNet-LSTM architecture
- **Video Detection**: Analyzes video sequences frame-by-frame to detect temporal inconsistencies and manipulation artifacts
- **Audio Detection**: Detects synthetic or cloned voices using Mel-Spectrogram analysis with CNN-based classification

The system provides real-time predictions with confidence scores, enabling users to verify media authenticity quickly and efficiently.

---

## 🏗️ System Architecture

The system follows a modular architecture with distinct pipelines for each media type:

1. **Input Layer**: User uploads media files (image/video/audio) through the Streamlit web interface
2. **Preprocessing Module**: Extracts features specific to each media type (face detection, frame extraction, Mel-Spectrogram generation)
3. **Detection Module**: Applies trained deep learning models for classification
4. **Output Layer**: Displays detection results with confidence scores and visual feedback

Each detection pipeline operates independently, allowing for scalable and maintainable code structure.

---

## 🔬 Methodology

### Image Detection

- **Architecture**: Hybrid MobileNet-LSTM model
- **Process**:
  - Face detection using Haar Cascade or MTCNN
  - Feature extraction using MobileNet (pre-trained on ImageNet)
  - LSTM layers to capture sequential dependencies in facial features
  - Binary classification (Real/Fake)
- **Input**: Single image file (JPG, PNG)
- **Output**: Classification label and confidence score

### Video Detection

- **Architecture**: Frame-based MobileNet-LSTM analysis
- **Process**:
  - Video is decomposed into individual frames using OpenCV
  - Each frame is processed through the image detection pipeline
  - Temporal analysis across frames to detect inconsistencies
  - Aggregated prediction based on frame-level results
- **Input**: Video file (MP4, AVI)
- **Output**: Overall classification with frame-wise analysis

### Audio Detection

- **Architecture**: CNN-based spectrogram classifier
- **Process**:
  - Audio preprocessing and noise reduction using Librosa
  - Mel-Spectrogram generation to convert audio into visual representation
  - CNN model trained on spectrogram features
  - Binary classification (Real/Synthetic)
- **Input**: Audio file (WAV, MP3)
- **Output**: Classification label and confidence score

---

## 🛠️ Technologies Used

### Programming Languages & Frameworks
- Python 3.8+
- TensorFlow / Keras
- Streamlit

### Libraries & Tools
- **Computer Vision**: OpenCV, PIL
- **Audio Processing**: Librosa
- **Data Processing**: NumPy, Pandas
- **Deep Learning**: TensorFlow, Keras
- **Visualization**: Matplotlib

### Models
- MobileNet (pre-trained CNN for feature extraction)
- LSTM (Long Short-Term Memory networks)
- Custom CNN architectures

---

## 📊 Dataset Information

The models were trained on publicly available deepfake detection datasets:

- **Image/Video Datasets**: FaceForensics++, Celeb-DF, DFDC (Deepfake Detection Challenge)
- **Audio Datasets**: ASVspoof, FoR (Fake-or-Real) dataset

**Note**: Due to size constraints and licensing restrictions, datasets are not included in this repository. Users must download datasets independently for model training or replication.

---

## 📈 Results and Performance Summary

The trained models demonstrate strong performance across all three modalities:

- **Image Detection**: High accuracy in identifying face-swapped and GAN-generated images
- **Video Detection**: Effective frame-level analysis with temporal consistency checks
- **Audio Detection**: Reliable classification of synthetic voice samples

Detailed performance metrics, including accuracy, precision, recall, and F1-scores, were evaluated during model training and validation phases.

---

## 🌐 Web Application Features

- **User-Friendly Interface**: Clean and intuitive design built with Streamlit
- **Multi-Format Support**: Accepts various image, video, and audio formats
- **Real-Time Detection**: Instant predictions with processing status indicators
- **Confidence Scores**: Transparent probability-based results
- **Visual Feedback**: Displays uploaded media alongside detection results
- **Modular Design**: Separate detection modules for each media type

---

## 👥 Team Members

This project was developed as a collaborative group effort:

- **Member 1**: [Name] - Model Development & Training
- **Member 2**: [Name] - Web Application & Integration
- **Member 3**: [Name] - Data Preprocessing & Testing
- **Member 4**: [Name] - Documentation & Deployment

*(Update with actual team member names and roles)*

---

## 🚀 How to Run the Project

### Prerequisites
- Python 3.8 or higher
- pip package manager
- Virtual environment (recommended)

### Installation Steps

1. **Clone the repository**
```bash
   git clone https://github.com/your-username/DEEPFAKE-DET.git
   cd DEEPFAKE-DET
```

2. **Create and activate virtual environment**
```bash
   python -m venv env
   # On Windows
   env\Scripts\activate
   # On macOS/Linux
   source env/bin/activate
```

3. **Install required dependencies**
```bash
   pip install -r requirements.txt
```

4. **Download pre-trained models**
   - Place `best_mobilenet_lstm_model.keras` in the `Models/` directory
   - Place `deepfake_audio_detector.h5` in the `Models/` directory
   - *(Models are not included in the repository due to file size limitations)*

5. **Run the Streamlit application**
```bash
   streamlit run app.py
```

6. **Access the application**
   - Open your browser and navigate to `http://localhost:8501`
   - Upload media files and test the detection system

---

## 📁 Project Structure
```
DEEPFAKE-DET/
│
├── Models/
│   ├── best_mobilenet_lstm_model.keras    # Image/Video detection model (excluded)
│   └── deepfake_audio_detector.h5         # Audio detection model (excluded)
│
├── sample-data/                            # Sample test files (optional)
├── seed/                                   # Initial scripts and utilities
├── tests/                                  # Unit tests
│
├── app.py                                  # Main Streamlit application
├── deepfake-det.ipynb                      # Model training notebook
├── generate_test_set.py                    # Test dataset generator
├── test_runner.py                          # Testing utilities
├── requirements.txt                        # Python dependencies
│
└── README.md                               # Project documentation
```

---

## ⚠️ Important Notes

- **Model Files**: Pre-trained model files (`.keras`, `.h5`) are **not included** in this repository due to GitHub file size limitations. These must be trained separately or obtained from the project team.
  
- **Datasets**: Training and testing datasets are **not included** due to size and licensing restrictions. Refer to the Dataset Information section for sources.

- **Computational Requirements**: Model training requires GPU acceleration (CUDA-enabled GPU recommended). Inference can run on CPU but may be slower.

- **Dependency Versions**: Ensure compatibility between TensorFlow, Keras, and CUDA versions if using GPU.

---

## 🔮 Future Scope

The project has significant potential for enhancement and expansion:

- **Model Optimization**: Implement model quantization and pruning for faster inference
- **Additional Modalities**: Extend detection to text-based deepfakes and multimodal content
- **Real-Time Processing**: Develop live video stream analysis capabilities
- **Mobile Deployment**: Create mobile applications for on-device detection
- **Explainable AI**: Integrate attention mechanisms and visualization tools to explain detection decisions
- **API Development**: Build RESTful APIs for integration with third-party applications
- **Blockchain Integration**: Implement content verification and provenance tracking
- **Cross-Platform Support**: Expand to browser extensions and social media plugins

---

## 📄 License / Usage Disclaimer

This project is developed for **educational and research purposes** as part of an academic engineering program. The system is intended to demonstrate deepfake detection techniques and should not be used as the sole method for verifying media authenticity in critical applications.

**Disclaimer**: The accuracy of detection depends on training data quality and model limitations. False positives and false negatives may occur. Users should exercise judgment and use multiple verification methods for important decisions.

For academic citations or commercial usage inquiries, please contact the project team.

---

## 🤝 Acknowledgments

Special thanks to the open-source community and researchers whose datasets, pre-trained models, and libraries made this project possible.

---

**⭐ If you find this project useful, please consider giving it a star on GitHub!**
