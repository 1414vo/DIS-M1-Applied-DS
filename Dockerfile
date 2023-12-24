FROM continuumio/miniconda3

WORKDIR ./ivp24

COPY . .

RUN mkdir data

RUN apt-get update && apt-get install -y \
    git

RUN conda env update --file environment.yml --name base

RUN pre-commit install

EXPOSE 8888
