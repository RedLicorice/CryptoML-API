FROM ubuntu:20.04

RUN apt-get update -y && apt-get install -y python3-pip python3-dev git

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
COPY ./cryptoml_common /cryptoml_common
COPY ./cryptoml_api /cryptoml_api
COPY ./cryptoml /cryptoml

# Move to system root
WORKDIR /
# Copy app launcher
COPY ./app.py /app.py
# Copy configuration files
COPY ./.env /.env
COPY ./config-docker.yml /config.yml

ENTRYPOINT [ "python3" ]
CMD [ "app.py" ]