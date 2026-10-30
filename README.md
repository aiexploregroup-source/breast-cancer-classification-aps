<<<<<<< HEAD
# Breast Cancer Image Classification

A collaborative college project by **Swaroopa and Chowda Reddy N**, completed in **2025**, demonstrating image preprocessing and trained-model inference through a Streamlit web application.

[Live demo](https://breast-cancer-classification-kbctfhvtuetb8wz6yz8fdz.streamlit.app/) | [Repository](https://github.com/Chowdaa/breast-cancer-classification)

> **Educational demonstration only.** This project is not a validated medical device and must not be used for screening, diagnosis, treatment decisions, or reassurance about a person's health. The application's diagnostic-sounding messages should be interpreted only as model-output labels.

## Overview

The application accepts an uploaded image, preprocesses it, loads a saved TensorFlow/Keras model, and displays the predicted **Benign** or **Malignant** class. It demonstrates the deployment and inference portion of an academic machine-learning project.

The current repository contains the application and dependency list. Model-training code, dataset documentation, evaluation results, and the model file itself are not included. The application describes the model as a custom CNN; its detailed architecture cannot be established from the application source alone.

## Features

- Home, Upload & Predict, and About pages using Streamlit navigation.
- JPG, JPEG, and PNG uploads with an image preview and original dimensions.
- RGB conversion, resizing to 48 × 48 pixels, and normalization to the range 0–1.
- Batched input with shape `(1, 48, 48, 3)`.
- Download of `model.h5` from the Google Drive location configured in `app.py` when the local file is absent.
- Cached model loading using `st.cache_resource`.
- Predicted class and a displayed model score.

## Technology stack

| Component | Technology |
| --- | --- |
| Interface | Python, Streamlit |
| Model inference | TensorFlow / Keras |
| Array processing | NumPy |
| Image processing | Pillow |
| Model download | gdown |
| Hosted demonstration | Streamlit Community Cloud |

## Repository contents

- `app.py`: navigation, model loading, preprocessing, prediction, and result display.
- `requirements.txt`: Python dependencies.
- `README.md`: project documentation.

`model.h5` is downloaded at runtime and is not distributed in this repository.

## Run locally

```bash
git clone https://github.com/Chowdaa/breast-cancer-classification.git
cd breast-cancer-classification
python -m venv .venv
```

Activate the environment:

```bash
# Windows PowerShell
.venv\Scripts\Activate.ps1

# macOS / Linux
source .venv/bin/activate
```

Install dependencies and launch:

```bash
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Open the local URL printed by Streamlit. Initial startup requires access to the configured Google Drive model, unless a compatible `model.h5` is already present in the working directory. Dependency versions are currently unpinned; TensorFlow/Keras compatibility may depend on the model's original export environment.

## Inference workflow

1. Load the saved model, downloading it if needed.
2. Open an uploaded image with Pillow.
3. Convert the image to RGB and resize to 48 × 48.
4. Convert pixel values to `float32`, divide by 255, and add a batch dimension.
5. Call `model.predict`.
6. Use `argmax` to select the class: index 0 is Benign; index 1 is Malignant.
7. Display the maximum model output multiplied by 100 as the application's confidence score.

This decoding assumes a two-class output with the stated class ordering. Verify these assumptions against the original training pipeline before changing the model. A single sigmoid output would require different decoding. The displayed score is not a clinically calibrated probability.

## Academic context and attribution

This was a **collaborative college project**, and the application credits **Swaroopa & Chowdareddy**. This copy preserves that shared attribution. Repository hosting and Git commit authorship alone do not describe the complete division of academic work.

The repository was shared as a fork of the teammate's original work. Retain upstream history and contributor credit when presenting or extending it. Individual responsibilities for dataset preparation, model training, interface development, evaluation, and deployment should be documented only when confirmed by the team.

=======
# breast-cancer-classification-aps
>>>>>>> 03be6a0b89e71354682bc2396ede55b9fb894259
