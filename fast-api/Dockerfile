FROM python:3.12-slim

WORKDIR /app


# Instalación de dependencias del sistema (git incluido)
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    build-essential
# COPY ["Pipfile", "Pipfile.lock", "./"]

COPY ["requirements.txt", "."]
RUN pip install -r requirements.txt

EXPOSE 8000

COPY . .

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1


CMD ["uvicorn", "main:app", "--reload", "--host", "0.0.0.0", "--port", "8000"]
