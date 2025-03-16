FROM python:3.10-slim-buster

WORKDIR /app

COPY . /app

RUN pip install -r requirements.txt 

RUN pip uninstall -y pinecone-plugin-inference || true

RUN pip install --force-reinstall pinecone[grpc]

CMD ["python3", "app.py"]