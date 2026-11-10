from fastapi import FastAPI, HTTPException, UploadFile, File
import uvicorn
import torch
import torch.nn as nn
import io
import json

from torchvision import transforms, models
from PIL import Image


transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])


with open("classes.json", "r", encoding="utf-8") as f:
    classes = json.load(f)


app = FastAPI()

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


model = models.resnet18(weights=None)

model.fc = nn.Linear(
    model.fc.in_features,
    len(classes)
)

model.load_state_dict(
    torch.load("model (16).pth", map_location=device))

model.to(device)
model.eval()


@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    try:
        image_data = await file.read()

        if not image_data:
            raise HTTPException(
                status_code=400,
                detail="Файл не загружен"
            )

        img = Image.open(
            io.BytesIO(image_data)
        ).convert("RGB")

        img_tensor = transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            y_pred = model(img_tensor)

            probabilities = torch.softmax(y_pred, dim=1)

            confidence, predicted = torch.max(
                probabilities, dim=1
            )

        return {
            "result": classes[predicted.item()],
            "confidence": round(confidence.item() * 100, 2)
        }

    except HTTPException:
        raise

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


if __name__ == "__main__":uvicorn.run(app,host="127.0.0.1",port=8000)