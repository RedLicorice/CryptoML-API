FROM ubuntu:20.04

RUN apt-get update -y && apt-get install -y python3-pip python3-dev git

# Copy over the app
COPY ./app /app

# Merge all requirements.txt and install everything
WORKDIR /app
RUN cat /app/cryptoml_core/requirements.txt \
 /app/cryptoml/requirements.txt \
 /app/cryptoml_api/requirements.txt \
 > requirements.txt && pip3 install -r requirements.txt

# Copy configuration files
COPY ./.env /app/.env
COPY ./config-docker.yml /app/config.yml

# Allow mounting of dependencies from docker-compose
ENV PYTHONPATH "${PYTHONPATH}:/python-dependencies"

# Add a new user
RUN useradd -ms /bin/bash user
# Give ownership of the application files to the newly created user
RUN chown -R user:user /app
USER user
ENTRYPOINT python3 app.py