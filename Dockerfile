FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y git

COPY . .

RUN --mount=type=cache,target=/root/.cache/pip \
  pip install -r requirements.txt

CMD [ "python", "./src/metrics/clip_score.py" ]

