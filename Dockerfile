FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y git

COPY setup.sh requirements.txt .

RUN --mount=type=cache,target=/root/.cache/pip \
 chmod u+x ./setup.sh && ./setup.sh

COPY . .

#CMD [ "python", "main.py" ]
CMD [ "python", "src/metrics/clip_score.py" ]

