FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y git

COPY . .

RUN --mount=type=cache,target=/root/.cache/pip \
 chmod u+x ./setup.sh && ./setup.sh

CMD [ "python", "pipeline.py" ]
