# 🖼️ Image & Signal Processing Portfolio

![Status](https://img.shields.io/badge/Status-Active-success)
![Language](https://img.shields.io/badge/Language-Python-blue)
![Tools](https://img.shields.io/badge/Tools-OpenCV%20|%20NumPy%20|%20Matplotlib-lightgrey)

Welcome to my repository featuring 5 advanced projects in image, signal, and video processing. This portfolio demonstrates the application of classic computer vision algorithms as well as generative AI models to analyze, manipulate, and generate visual and audio data.

## 📂 Project Architecture

| Folder | Project Topic | Technical Description |
| :--- | :--- | :--- |
| 📁 **`ex1`** | **Video Scene Cut Detection** | Automatic detection of scene cuts using standard and cumulative histogram distances to accurately identify transitions and filter out false positives caused by gradual illumination changes. |
| 📁 **`ex_2`** | **Audio Watermarking & Analysis** | Manipulation of audio signals in the frequency domain (using the Fourier Transform) to embed, detect, and classify invisible Frequency Modulation (FM) watermarks. It also includes distinguishing between time-domain and frequency-domain speed modifications. |
| 📁 **`ex_3`** | **Image Blending & Hybrid Images** | Creation of seamless transitions (image blending) and hybrid images by manipulating spatial frequencies through the construction of Gaussian and Laplacian pyramids. |
| 📁 **`ex_4`** | **Stereo Mosaicing (Panoramas)** | Generation of panoramic views from panning videos through geometric alignment. This involves robust motion estimation using SIFT feature extraction, descriptor matching, RANSAC outlier rejection, and rigid transformation modeling. |
| 📁 **`ex5`** | **Generative AI & Forensics** | Exploration of Stable Diffusion as a generative prior for personalized image generation (DreamBooth/LoRA) and prompt-guided editing (SDEdit). It also features a custom scoring method for membership inference to detect which images were used in the model's fine-tuning dataset. |

## 🛠️ Technologies & Tools
* **Language:** Python
* **Vision & Algorithms:** OpenCV (`cv2`) for image/video processing and feature detection, and NumPy for fast numerical computations, matrix operations (SVD), and custom convolutions.
* **Visualization:** Matplotlib for analyzing spectrograms, histogram distances, and camera motion trajectories.
* **Deep Learning:** Diffusion architectures (Stable Diffusion) and fine-tuning techniques (LoRA/DreamBooth).

## 🚀 Installation & Usage

To test these projects locally, follow these steps:

```bash
# 1. Clone the repository
git clone [https://github.com/itzhakfagebaume/image_processing.git](https://github.com/itzhakfagebaume/image_processing.git)

# 2. Navigate to the desired project folder
cd image_processing/ex1

# 3. Ensure you have the required dependencies
pip install opencv-python numpy matplotlib
