FROM python:3.10.14-bullseye
WORKDIR /app
COPY . /app
RUN apt update -y
RUN pip install -r requirements.txt
EXPOSE 5000
CMD ["python3", "app.py"]

