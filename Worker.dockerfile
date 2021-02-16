FROM ubuntu:20.04

RUN apt-get update -y && apt-get install -y python3-pip python3-dev redis-tools git


# We copy just the requirements.txt first to leverage Docker cache
COPY ./cryptoml_api/requirements.txt /cryptoml_api/requirements.txt
COPY ./cryptoml/requirements.txt /cryptoml/requirements.txt

# Install API requirements
WORKDIR /cryptoml_api
RUN pip3 install -r requirements.txt

# Install LIB requirements
WORKDIR /cryptoml
RUN pip3 install -r requirements.txt

# Copy API and LIB sources
COPY ./cryptoml_api /cryptoml_api
COPY ./cryptoml /cryptoml

#Move to system root
WORKDIR /
# Copy worker launcher
COPY ./worker.py /worker.py
# Copy configuration files
COPY ./.env /.env
COPY ./config-docker.yml /config.yml

# Add a new user: celery workers should not be run as root!
RUN useradd -ms /bin/bash worker

# Give ownership of the application files to the newly created user
RUN chown -R worker:worker /cryptoml_api
RUN chown -R worker:worker /cryptoml
RUN chown worker:worker /worker.py
RUN chown worker:worker /config.yml
RUN chown worker:worker /.env

USER worker

ENTRYPOINT celery -A worker.celery worker --loglevel=info