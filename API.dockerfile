FROM ubuntu:20.04

RUN apt-get update -y && \
    apt-get install -y python3-pip python3-dev

# We copy just the requirements.txt first to leverage Docker cache
COPY ./api/requirements.txt /api/requirements.txt

WORKDIR /api

RUN pip3 install -r requirements.txt

COPY ./api /api

WORKDIR /
COPY ./app.py /app.py
# Use a different config file for docker environment
COPY ./.env /.env
COPY ./config-docker.yml /config.yml

ENTRYPOINT [ "python3" ]
CMD [ "app.py" ]