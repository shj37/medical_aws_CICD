FROM python:3.10-slim-buster

WORKDIR /app

COPY . /app

RUN pip install -r requirements.txt --no-cache-dir

RUN pip uninstall -y pinecone-plugin-inference --no-cache-dir

CMD ["python3", "app.py"]