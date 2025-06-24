import streamlit as st
from streamlit_drawable_canvas import st_canvas
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
import psycopg2
from datetime import datetime


# loading model class
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 16, 3, 1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, 1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(32 * 5 * 5, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        return self.model(x)


# load model
device = torch.device("cpu")
model = CNN()
model.load_state_dict(torch.load("mnist_cnn.pt", map_location=device))
model.eval()

# canvas
st.title("MNIST Based Digit Classifier")
canvas = st_canvas(
    fill_color="white",
    stroke_width=10,
    stroke_color="white",
    height=280,
    width=280,
    drawing_mode="freedraw",
    key="canvas",
)

if canvas.image_data is not None:
    img = Image.fromarray((255 - canvas.image_data[:, :, 0]).astype(np.uint8))
    img = img.resize((28, 28)).convert('L')
    img_arr = np.array(img).astype(np.float32) / 255.0
    tensor = torch.tensor(img_arr).unsqueeze(0).unsqueeze(0)

    if st.button("Predict"):
        with torch.no_grad():
            output = model(tensor)
            prob = torch.softmax(output, dim=1)
            pred = prob.argmax().item()
            confidence = prob[0, pred].item()

        st.write(f"Prediction: **{pred}**")
        st.write(f"Confidence: **{confidence*100:.2f}**")

        true_label = st.number_input("What is the true digit?", min_value=0, max_value=9, step=1)
        if st.button("Log Result"):
            # Log to PostgreSQL
            try:
                conn = psycopg2.connect(
                    host="db", dbname="mnist_db", user="user", password="password"
                )
                cur = conn.cursor()
                cur.execute("""
                    INSERT INTO logs (timestamp, prediction, confidence, true_label)
                    VALUES (%s, %s, %s, %s)
                """, (datetime.now(), pred, confidence, true_label))
                conn.commit()
                cur.close()
                conn.close()
                st.success("Logged!")
            except Exception as e:
                st.error(f"Error logging: {e}")