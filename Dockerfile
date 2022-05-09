# Copyright 2021 Allvision IO, Inc.
# author: ryan@allvision.io
# usage: docker build -t <tag> -f <path/to/this/file>
FROM python:3.7.13-slim

ARG DEBIAN_FRONTEND=noninteractive

RUN python --version && pip --version

RUN apt update \
 && apt-get install -y git nano

WORKDIR /code

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

COPY ./src/ /code
