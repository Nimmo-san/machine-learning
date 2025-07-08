import os
from datetime import datetime
import numpy as np
import psycopg2
import streamlit as st
from streamlit_drawable_canvas import st_canvas
import torch
import torch.nn as nn
from PIL import Image, ImageOps


DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "mnist")
DB_USER = os.getenv("DB_USER", "mnist")
DB_PASSWORD = os.getenv("DB_PASSWORD", "mnistpass")
MODEL_PATH = os.getenv("MODEL_PATH", "mnist_cnn.pt")


# loading model class
class CNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(1, 6, 3),
            nn.ReLU(),
            nn.BatchNorm2d(6),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(6, 16, 3),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.out = nn.Sequential(
            nn.Flatten(),
            nn.LazyLinear(128),
            nn.ReLU(),
            nn.Linear(128, 10),
            nn.Softmax(dim=-1),
        )

    def forward(self, X):
        X = self.cnn(X)
        X = self.out(X)
        return X


def get_connection():
    conn = psycopg2.connect(
        host=DB_HOST, port=DB_PORT, dbname=DB_NAME, user=DB_USER, password=DB_PASSWORD
    )

    with conn.cursor() as cur:
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS logs (
                id SERIAL PRIMARY KEY,
                timestamp TIMESTAMP,
                prediction SMALLINT NOT NULL,
                confidence FLOAT NOT NULL,
                true_label SMALLINT,
            );
            """
        )
        return conn

def load_model():
    model = CNN()
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()
    return model


model = load_model()
conn = get_connection()

st.set_page_config(page_title="MNIST Digit Classifier", layout="centered")
st.title("MNIST Digit Classifier")
st.markdown("Draw a digit(0-9) in the box, and click predict to see the result.")

canvas = st_canvas(
    fill_color="black",
    stroke_color="white",
    stroke_width=10,
    width=280,
    height=280,
    drawing_mode="freedraw",
    key="canvas"
)

col1, col2 = st.columns(2)
pred_button = col1.button("Predict")

if canvas.image_data is not None:
    if pred_button:
        # Convert RGBA image_data -> grayscale 28×28
        img = Image.fromarray(canvas.image_data.astype("uint8"), "RGBA")
        img = img.convert("L")  # grayscale
        img = ImageOps.invert(img)
        img = img.resize((28, 28))
        img_np = np.array(img, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(img_np).unsqueeze(0).unsqueeze(0)
        with torch.no_grad():
            logits = model(tensor)
            probs = torch.softmax(logits, dim=1)[0]
            pred = int(torch.argmax(probs))
            conf = float(probs[pred])
        st.subheader(f"Prediction: {pred}")
        st.write(f"Confidence: {conf:.2%}")

        true_label = st.number_input("True label (optional)", min_value=0, max_value=9, step=1, format="%d")
        if st.button("Submit true label"):
            print(pred, f"{conf:.2f}", true_label)
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO predictions (predicted, confidence, true_label) VALUES (%s, %s, %s)",
                    (pred, conf, int(true_label) if true_label is not None else None),
                )
            st.success("Logged to database!")