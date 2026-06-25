from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from google.cloud import storage
import joblib
import os

app = FastAPI()

GCS_BUCKET = os.environ.get("GCS_BUCKET", "day21-ai20k-viethoang5426")
GCS_MODEL_KEY = "models/latest/model.pkl"
MODEL_PATH = os.path.expanduser("~/models/model.pkl")

def download_model():
    """Tải file model.pkl từ GCS về máy khi server khởi động."""
    try:
        client = storage.Client()
        bucket = client.bucket(GCS_BUCKET)
        blob = bucket.blob(GCS_MODEL_KEY)
        blob.download_to_filename(MODEL_PATH)
        print("Đã tải mô hình thành công từ GCS.")
    except Exception as e:
        print(f"Chưa thể tải mô hình (có thể là lần đầu chạy): {e}")

download_model()
try:
    model = joblib.load(MODEL_PATH)
except:
    model = None

class PredictRequest(BaseModel):
    features: list[float]

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(req: PredictRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Mô hình chưa sẵn sàng.")
    if len(req.features) != 12:
        raise HTTPException(status_code=400, detail="Cần cung cấp đúng 12 đặc trưng.")
    
    preds = model.predict([req.features])
    pred = int(preds[0])
    label_map = {0: "thấp", 1: "trung_bình", 2: "cao"}
    
    return {"prediction": pred, "label": label_map.get(pred, "unknown")}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
