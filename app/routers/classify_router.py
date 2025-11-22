from fastapi import APIRouter, File, UploadFile, HTTPException
import numpy as np
from app.models.mobilenet import model
from app.schemas.predict import Response, TopPrediction
import tensorflow as tf
from tf_keras.applications.mobilenet_v2 import preprocess_input
import json
import os

router = APIRouter(prefix="/classify", tags=["classify"])

ALLOWED_FORMATS = ["image/jpeg", "image/png", "image/jpg"]
ALLOWED_SIZE = 1024 * 1024 *2

CLASS_INDEX_PATH = os.path.join("app", "static", "imagenet_class_index.json")
with open(CLASS_INDEX_PATH, "r") as f:
    CLASS_INDEX = json.load(f)

def decode_predictions_local(preds, top=5):
    # preds: shape (1000,)
    top_indices = preds.argsort()[-top:][::-1]
    results = [
        (i, CLASS_INDEX[str(i)][1], float(preds[i]))
        for i in top_indices
    ]
    return results


@router.post("/",response_model=Response )
async def classify(file: UploadFile = File(..., description="Upload file", example=b"image/jpeg")):
    try:
        if file.content_type not in ALLOWED_FORMATS:
            raise HTTPException(
                status_code=400,
                detail=f"Image format not supported. allowed formats: {ALLOWED_FORMATS}"
            )

        img_bytes = await file.read()

        if len(img_bytes) > ALLOWED_SIZE:
            raise HTTPException(
                status_code=400,
                detail="Image size exceeds 2mb."
            )

        try:
            img = tf.io.decode_image(img_bytes, channels=3)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Cannot decode image: {str(e)}")

        img = tf.image.resize(img, (224, 224))
        img = np.array(img, dtype=np.float32)

        img = preprocess_input(img)
        img = np.expand_dims(img, axis=0)

        preds = model.predict(img)
        if preds.ndim == 2 and preds.shape[0] == 1:
            preds = preds[0]

        results = decode_predictions_local(preds, top=5)
        best_pred = results[0]
        prediction_name = best_pred[1].replace("_", " ")
        confidence = float(best_pred[2])
        top_5 = [
            TopPrediction(
                rank=i + 1,
                name=label.replace("_", " "),
                confidence=float(prob)
            )
            for i, (_, label, prob) in enumerate(results)
        ]

        return Response(
            predicted_name_image=prediction_name,
            confidence=confidence,
            probability= top_5
        )

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=400, detail=f"Processing failed: {str(e)}")