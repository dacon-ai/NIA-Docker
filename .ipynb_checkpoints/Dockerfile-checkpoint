ARG TENSORFLOW="2.3.1-gpu"

FROM tensorflow/tensorflow:${TENSORFLOW}

RUN apt-get update && apt-get install -y git

RUN git clone https://github.com/dacon-ai/NIA-Docker.git

WORKDIR /NIA-Docker

RUN python -m pip install --upgrade pip

RUN python -m pip install -r requirements.txt
