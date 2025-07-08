FROM python:3.9-slim-bullseye

RUN apt-get update && \
	apt-get upgrade -y && \
	apt-get dist-upgrade -y && \
	apt-get autoremove -y && \
	apt-get clean && \
	rm -rf /var/lib/apt/lists/*


WORKDIR /app
COPY . .

# RUN pip install --no-cache-dir torch torchvision streamlit streamlit_drawable_canvas numpy psycopg2-binary
RUN pip install -r requirements.txt

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]