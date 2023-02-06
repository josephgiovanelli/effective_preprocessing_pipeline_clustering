FROM mcr.microsoft.com/vscode/devcontainers/python:0-3.6
COPY . /home/effective_preprocessing_pipeline_clustering
WORKDIR /home/effective_preprocessing_pipeline_clustering
RUN pip install --upgrade pip && \
    pip install -r requirements.txt 