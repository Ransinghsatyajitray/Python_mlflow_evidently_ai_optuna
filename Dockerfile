FROM python:3.10.14-bullseye
WORKDIR /service
COPY requirements.txt .
COPY . ./
RUN pip install -r requirements.txt
ENTRYPOINT ["python3", "app.py"]

