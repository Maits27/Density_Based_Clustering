FROM python:3.10.13

COPY . /app/

WORKDIR /app

RUN apt update

RUN pip install --upgrade pip

RUN pip install -r requirements.txt

RUN python -m spacy download en_core_web_sm

WORKDIR /app/src

ENTRYPOINT ["python", "main.py"]
