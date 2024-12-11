FROM python:3.9-slim

RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1

RUN pip install --no-cache-dir flask opencv-python-headless numpy ultralytics

COPY . /app

WORKDIR /app

EXPOSE 8080

CMD ["python", "app.py"]

