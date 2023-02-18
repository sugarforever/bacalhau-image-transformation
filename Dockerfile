FROM python:3.8-slim

WORKDIR /app

COPY requirements.txt /app

RUN pip3 install -r requirements.txt
RUN pip3 install pillow
COPY image_transformation.py /app/

ENTRYPOINT ["python3", "image_transformation.py", "/app/inputs/", "/app/outputs/"]