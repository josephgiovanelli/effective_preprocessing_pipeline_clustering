FROM mcr.microsoft.com/vscode/devcontainers/python:0-3.6
RUN cd home && mkdir autoclues
WORKDIR /home/autoclues
COPY . .
RUN pip install --upgrade pip && \
    pip install -r requirements.txt
RUN mkdir results
RUN chmod 777 scripts/*
ENTRYPOINT ["./scripts/wrapper_experiments.sh"]