# Based on the tutorial on 
# https://docs.docker.com/language/python/build-images/
FROM python:3.8-slim-buster

# Create the working directory
WORKDIR /app

# Install Python dependencies
COPY requirements.txt requirements.txt
RUN pip install --upgrade pip~=21.1.3
RUN pip install -r requirements.txt
COPY . .

ENV PYTHONPATH /app
# CMD flask run --host 0.0.0.0