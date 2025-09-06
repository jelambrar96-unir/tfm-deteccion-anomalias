FROM python:3.12-slim

WORKDIR /app

# COPY ["Pipfile", "Pipfile.lock", "./"]

COPY ["requirements.txt", "."]
RUN pip install -r requirements.txt

COPY . .

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1


EXPOSE 8888

CMD jupyter notebook \
    --ip=0.0.0.0 \
    --port=8888 \
    --no-browser \
    --allow-root \
    --IdentityProvider.token=${JUPYTER_TOKEN}
