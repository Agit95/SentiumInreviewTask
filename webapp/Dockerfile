FROM python:3.7.5-slim

WORKDIR /python-docker

RUN mkdir model
COPY model/london.h5 ./model/london.h5
COPY model/model_columns.json ./model/model_columns.json
COPY app.py ./app.py

CMD ["python3" "-m" "flask", "run"]