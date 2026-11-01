# 🖼️ ImageGenLocal

A fast, local platform for AI image creation.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Status](https://img.shields.io/badge/Status-Development-blue.svg)]()

## 💡 Project Overview

This project provides a fully functional, local environment to generate high-quality images using local models setup and used from HuggingFace. It allows users to generate content instantly without requiring external API calls, offering speed, privacy, and complete control over the generation process. 

**Core Goal:** To democratize AI image generation by making powerful models accessible directly on local hardware.

---

## 🚀 Tech Stack

*   **Model:** `[Model Name]` (e.g., Stable Diffusion)
*   **Framework:** `[PyTorch/TensorFlow]`
*   **Libraries:** `[OpenCV, PIL, Transformers, etc.]`
*   **Interface:** `[Gradio/Streamlit]`

## ⚙️ Setup Guide

Follow these steps to get the project running locally.

### Prerequisites

*   Python 3.x
*   NVIDIA GPU (Recommended for performance)
*   Git

### Installation

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/Asutosh11/ImageGenLocal.git
    cd ImageGenLocal
    ```
2.  **Create and Activate Environment:**
    Use a virtual environment for clean dependencies.
    ```bash
    python -m venv venv
    .env\Scriptsctivate  # On Windows
    source venv/bin/activate  # On Linux/macOS
    ```
3.  **Install Dependencies:**
    Install all required libraries from the `requirements.txt` file:
    ```bash
    pip install -r requirements.txt
    ```
4.  **Configuration:**
    Edit the configuration file (e.g., `config.py`) to set up your model paths and parameters.

### Running the Script

Execute the main Python script to start the application:
```bash
python generate.py
```

---

## 🖥️ Hardware & Performance

### ⚙️ My Current Setup

| Component | Detail |
| :--- | :--- |
| **GPU** | **NVIDIA GeForce GTX 1660** |
| **VRAM** | **6 GB** |
| **CPU** | **11th Gen Intel(R) Core(TM) i5-11600 @ 2.80GHz (2.81 GHz)** |
| **RAM** | **16.0 GB** |

### ⬆️ Upgrade Path

*   **Upgrade Target:** RTX 3060 (12GB VRAM) or higher.
*   **Benefit:** Enables faster generation and higher resolutions by increasing VRAM capacity.

## 📸 Showcase

**Visual Examples**

## Tools

<p align="center">
  <img src="./screenshots/the_tool/t2.png" />
</p>
<p align="center">
  <img src="./screenshots/the_tool/t1.png" />
</p>
<p align="center">
  <img src="./screenshots/the_tool/t3.png" />
</p>

## Image Generations

<p align="center">
  <img src="./screenshots/generated/g3.png" width="48%" />
  <img src="./screenshots/generated/g4.png" width="48%" />
</p>

<p align="center">
  <img src="./screenshots/generated/g8.png" width="48%" />
  <img src="./screenshots/generated/g5.png" width="48%" />
</p>

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).
