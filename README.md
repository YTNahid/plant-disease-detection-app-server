# 🌿 Plant Disease Detection API

A FastAPI backend that classifies plant diseases from leaf images using a MobileNet deep learning model trained on the PlantVillage dataset. Upload a photo of a plant leaf and get back the disease name, confidence score, description, treatment, and prevention advice.

---

## Features

- Detects **38 plant disease classes** across 14 crops including Tomato, Potato, Apple, Corn, Grape, and more
- Returns **disease description, treatment, and prevention** for each prediction
- **Batch prediction** — classify up to 10 images in a single request
- Confidence threshold — rejects low-confidence predictions instead of guessing
- Image validation — file type, size, and format checks
- Clean, consistent error responses

---

## Dataset

Trained on the **[New Plant Diseases Dataset](https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset)** — an augmented version of the original [PlantVillage Dataset](https://www.kaggle.com/datasets/emmarex/plantdisease) available on Kaggle.

This dataset consists of about 87K rgb images of healthy and diseased crop leaves which is categorized into 38 different classes. The total dataset is divided into 80/20 ratio of training and validation set preserving the directory structure. A new directory containing 33 test images is created later for prediction purpose.

---

## Tech Stack

- **FastAPI** — REST API framework
- **TensorFlow / Keras** — model inference
- **OpenCV** — image preprocessing
- **Uvicorn** — ASGI server

---

## API Endpoints

### Inference

| Method | Endpoint                | Description                      |
| ------ | ----------------------- | -------------------------------- |
| `POST` | `/api/v1/predict`       | Classify a single leaf image     |
| `POST` | `/api/v1/predict/batch` | Classify up to 10 images at once |

---

### Model

| Method | Endpoint             | Description                                             |
| ------ | -------------------- | ------------------------------------------------------- |
| `GET`  | `/api/v1/model/info` | Model metadata (input shape, output shape, param count) |
| `GET`  | `/api/v1/classes`    | List all 38 class labels with their index               |

---

### Health

| Method | Endpoint  | Description                         |
| ------ | --------- | ----------------------------------- |
| `GET`  | `/`       | API status check                    |
| `GET`  | `/health` | Health check with model load status |

---

## Supported Crops & Diseases

Apple, Blueberry, Cherry, Corn, Grape, Orange, Peach, Bell Pepper, Potato, Raspberry, Soybean, Squash, Strawberry, Tomato — **38 classes total** including healthy variants.
