from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi import Request
from PIL import Image
import io

from model_loader import predict_image
# from utils import galaxy_info

app = FastAPI()
app.mount("/js", StaticFiles(directory="frontend/js"), name="js")
app.mount("/3d-models", StaticFiles(directory="frontend/3d-models"), name="models")

templates = Jinja2Templates(directory="frontend")

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")

    label, confidence = predict_image(image)

    return {
        "prediction": label,
        "confidence": round(confidence * 100, 2)
    }
# galaxy_info[label]