FROM python:3.8-slim

WORKDIR /app

COPY requirements.txt /app

RUN pip3 install -r requirements.txt

COPY image_transformation.py /app/

ENTRYPOINT ["python3", "image_transformation.py"]