FROM ubuntu:20.04

RUN apt-get update -y && \
    apt-get install -y python3-pip python3-dev

# We copy just the requirements.txt first to leverage Docker cache
COPY ./src/requirements.txt /src/requirements.txt

WORKDIR /src

RUN pip3 install -r requirements.txt

COPY ./src /src

WORKDIR /
COPY ./app.py /app.py
# Use a different config file for docker environment
COPY ./.env /.env
COPY ./config-docker.yml /config.yml

ENTRYPOINT [ "python3" ]
CMD [ "app.py" ]