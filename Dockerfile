FROM python:3.8.6-buster

COPY api /api
COPY SolarVision /SolarVision
COPY sv_model.h5 /sv_model.h5
COPY seg_model_weights.h5 /seg_model_weights.h5
COPY requirements.txt /requirements.txt
COPY key2.json /credentials.json

RUN pip install -r requirements.txt
ENV GOOGLE_APPLICATION_CREDENTIALS=credentials.json

CMD uvicorn api.api:app --host 0.0.0.0 --port $PORT