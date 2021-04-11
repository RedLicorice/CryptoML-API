FROM ubuntu:20.04

RUN apt-get update -y && apt-get install -y python3-pip python3-dev redis-tools git

# We copy just the requirements.txt first to leverage Docker cache
COPY ./cryptoml_core/requirements.txt /app/cryptoml_core/requirements.txt
COPY ./cryptoml/requirements.txt /app/cryptoml/requirements.txt
COPY ./cryptoml_api/requirements.txt /app/cryptoml_api/requirements.txt

# Install LIB requirements
WORKDIR /app/cryptoml
RUN pip3 install -r requirements.txt

# Install CORE requirements
WORKDIR /app/cryptoml_core
RUN pip3 install -r requirements.txt

# Install API requirements
WORKDIR /app/cryptoml_api
RUN pip3 install -r requirements.txt

# Copy app sources
COPY ./cryptoml_core /app/cryptoml_core
COPY ./cryptoml_api /app/cryptoml_api
COPY ./cryptoml /app/cryptoml
COPY ./app.py /app/app.py
COPY ./worker.py /app/worker.py
# Copy configuration files
COPY ./.env /app/.env
COPY ./config-docker.yml /app/config.yml

# Allow mounting of dependencies from docker-compose
ENV PYTHONPATH "${PYTHONPATH}:/python-dependencies"

# Add a new user: celery workers should not be run as root!
RUN useradd -ms /bin/bash worker
# Give ownership of the application files to the newly created user
RUN chown -R worker:worker /app
USER worker
WORKDIR /app
ENTRYPOINT celery -A worker.celery worker --loglevel=info