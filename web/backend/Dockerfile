# The recognition and translation API.
#
# Build from the REPOSITORY ROOT, not from this directory:
#
#   docker build -f web/backend/Dockerfile -t isuara-api .
#
# The context has to include app/src/main/assets, because config.py reads the
# model and the label map straight out of the Android app's assets rather than
# keeping a second copy — that is what stops the phone and the web build ever
# drifting onto different models.
#
# python:3.12-slim, not the TensorFlow image: the shipped model is Flex-free, so
# requirements.txt installs the 48MB LiteRT interpreter instead of TensorFlow's
# 1.4GB. That is what lets this fit a free host.

FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /srv

# Dependencies first, so a code change does not reinstall them.
COPY web/backend/requirements.txt web/backend/requirements.txt
RUN pip install --no-cache-dir -r web/backend/requirements.txt

COPY web/backend/app web/backend/app

# Only the two files the server actually reads. Copying the whole assets
# directory would drag in the MediaPipe .task bundles and the older Flex
# models — about 30MB the API never opens, since the browser loads those from
# the frontend's own origin.
COPY app/src/main/assets/bim_lstm_v312_fp16.tflite app/src/main/assets/
COPY app/src/main/assets/label_map.json app/src/main/assets/

WORKDIR /srv/web/backend

# Fail the build if the model cannot load, rather than starting a server that
# answers every prediction with an error.
RUN python -c "from app.inference import SignInterpreter; i = SignInterpreter(); print('model ok:', i.backend)"

EXPOSE 8000

# $PORT is assigned by the host (Render, Fly, Railway all set it); 8000 locally.
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
