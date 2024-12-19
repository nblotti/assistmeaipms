FROM python:3.12-alpine

WORKDIR /app

# set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
ENV ENVIRONNEMENT PROD



# install dependencies
COPY requirements.txt /app/requirements.txt

RUN pip install --upgrade pip
RUN apk update && \
 apk add --no-cache --virtual .build-deps gcc python3-dev musl-dev && \
 python3 -m pip install -r requirements.txt --no-cache-dir && \
 apk --purge del .build-deps




# copy project
COPY . .
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
