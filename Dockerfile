FROM python:3.10.13

COPY . /app/

WORKDIR /app

RUN apt update

RUN pip install --upgrade pip

RUN pip install -r requirements.txt

WORKDIR /app/src

ENTRYPOINT ["python", "main.py"]
