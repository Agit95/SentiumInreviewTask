FROM python:3.7.5-slim

WORKDIR .

COPY application.py ./application.py
COPY data_processor.py ./data_processor.py
COPY models.py ./models.py
COPY trainer.py ./trainer.py
COPY main.py ./main.py

CMD ["python3", "main.py"]