import streamlit as st
import torch
import torch.nn as nn
import json

from torchvision import transforms, models
from PIL import Image


st.set_page_config(
    page_title="Sport Classification",
    page_icon="",
    layout="centered"
)


device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


with open("classes.json", "r", encoding="utf-8") as f:
    classes = json.load(f)


@st.cache_resource
def load_model():

    model = models.resnet18(weights=None)

    model.fc = nn.Linear(
        model.fc.in_features,
        len(classes)
    )

    model.load_state_dict(
        torch.load("model (16).pth", map_location=device)
    )

    model.to(device)
    model.eval()

    return model


model = load_model()


transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])


st.title("Sport Classification")

st.write(
    "Загрузите изображение, чтобы определить вид спорта."
)


uploaded_file = st.file_uploader(
    "Выберите изображение",
    type=["jpg", "jpeg", "png"]
)


if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Загруженное изображение",
        use_container_width=True
    )

    if st.button("Распознать"):

        image_tensor = transform(image)
        image_tensor = image_tensor.unsqueeze(0).to(device)

        with torch.no_grad():

            outputs = model(image_tensor)

            probabilities = torch.softmax(outputs, dim=1)

            confidence, predicted = torch.max(
                probabilities, dim=1
            )

        result = classes[predicted.item()]

        st.success(f"Вид спорта: {result}")

        st.metric(
            "Уверенность модели",
            f"{confidence.item() * 100:.2f}%"
        )
