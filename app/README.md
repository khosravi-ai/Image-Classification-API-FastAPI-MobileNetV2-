# Image Classification API (FastAPI + MobileNetV2)

## 📌 Overview
This project provides a FastAPI-based web service that classifies uploaded images using a pre-trained **MobileNetV2** deep learning model.  
Users can send an image (JPG/PNG), and the API returns:

- Predicted class name  
- Confidence score  
- Top-5 predictions with probabilities

The model is stored locally and runs offline without external dependencies.

---

## 🚀 Features
- FastAPI backend  
- Pretrained MobileNetV2 (`.keras` file)  
- Manual ImageNet class index loading (no internet required)  
- Handles image validation (format + size)
- Swagger API documentation built in

---

## 📂 Project Structure
```
project/
│
├── app/
│   ├── main.py               # FastAPI app entrypoint
│   ├── models/
│   │   └── mobilenet.py      # Loads MobileNetV2 model
│   ├── routers/
│   │   └── classify_router.py # API endpoint
│   ├── schemas/
│   │   └── predict.py        # Response models (Pydantic)
│   ├── static/
│   │   └── imagenet_class_index.json
│
├── mobilenet_v2_full.keras   # Saved pretrained model
├── requirements.txt
└── README.md
```

---

## 📦 Installation

### 1️⃣ Create Virtual Environment
```bash
python -m venv .venv
```

### 2️⃣ Activate Environment
**Windows**
```bash
.\.venv\Scripts\activate
```

**Linux/Mac**
```bash
source .venv/bin/activate
```

### 3️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

---

## ▶️ Run Server
```bash
uvicorn app.main:app --reload
```

Server runs at:

```
http://127.0.0.1:8000
```

API docs (Swagger):

```
http://127.0.0.1:8000/docs
```

---

## 📤 Making a Request

### cURL Example
```bash
curl -X POST "http://127.0.0.1:8000/classify/" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@image.jpg"
```

### Response Example
```json
{
  "predicted_name_image": "Persian cat",
  "confidence": 0.92,
  "probability": [
    { "rank": 1, "name": "Persian cat", "confidence": 0.92 },
    { "rank": 2, "name": "Siamese cat", "confidence": 0.04 }
  ]
}
```

---

## ⚠️ Supported Formats
| Format | Status |
|--------|--------|
| JPG    | ✔ |
| JPEG   | ✔ |
| PNG    | ✔ |

Max file size: **2 MB**

---

## 🧠 Model
- Pretrained MobileNetV2 from TensorFlow
- Converted and saved locally:

```python
from tensorflow.keras.applications import MobileNetV2
model = MobileNetV2(weights='imagenet')
model.save("mobilenet_v2_full.keras")
```

---

## 📎 ImageNet Class Index
Since `decode_predictions` may require internet access, we manually download:

```
imagenet_class_index.json
```

and load it locally for offline usage.

---

## 📘 License
This project is built for educational and personal development purposes.  
Feel free to extend or use it commercially.

---

## 🙌 Author
Hossein Khosravi

---
Enjoy using the API!
