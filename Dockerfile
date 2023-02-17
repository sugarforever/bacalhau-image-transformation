FROM python:3.8-slim

COPY image_transformation.py /app/

WORKDIR /app

COPY requirements.txt /app

RUN pip3 install -r requirements.txt

ENTRYPOINT ["python3", "image_transformation.py"]