FROM python:3.10-slim-buster

WORKDIR /app

COPY . /app

RUN pip install -r requirements.txt && pip uninstall -y pinecone-plugin-inference

CMD ["python3", "app.py"]