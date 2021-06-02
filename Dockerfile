FROM python:3.8.6-buster

COPY api /api
COPY SolarVision /SolarVision
COPY sv_model.h5 /sv_model.h5
COPY requirements.txt /requirements.txt

RUN pip install -r requirements.txt

CMD uvicorn api.api:app --host 0.0.0.0 --port $PORT