FROM ubuntu:20.04

#RUN apt-get update -y && apt-get install -y python3-pip python3-dev git
ENV PYTHONUNBUFFERED=1

RUN echo "**** install Python ****" && \
    apt-get update -y && \
    apt-get install -y python3-pip python3-dev wget git && \
    if [ ! -e /usr/bin/python ]; then ln -sf python3 /usr/bin/python ; fi && \
    echo "**** install pip ****" && \
    pip3 install --no-cache --upgrade pip setuptools wheel && \
    if [ ! -e /usr/bin/pip ]; then ln -s pip3 /usr/bin/pip ; fi

# TA-Lib
RUN echo "**** install TA-Lib ****" && \
    wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz && \
    tar -xvzf ta-lib-0.4.0-src.tar.gz && \
    cd ta-lib/ && \
    ./configure --prefix=/usr && \
    make && \
    make install

# Add the requirements first to speed up further builds (leveraging Docker cache)
# Separate layers so caching acts nicely and only one layer is rebuilt if requirements
# change
ADD ./app/cryptoml/requirements.txt /app/cryptoml/requirements.txt
RUN echo "**** install CryptoML ****" && cd /app/cryptoml && pip3 install -r requirements.txt

ADD ./app/cryptoml_core/requirements.txt /app/cryptoml_core/requirements.txt
RUN echo "**** install CryptoML-Core ****" && cd /app/cryptoml_core && pip3 install -r requirements.txt

ADD ./app/cryptoml_api/requirements.txt /app/cryptoml_api/requirements.txt
RUN echo "**** install CryptoML-API ****" && cd /app/cryptoml_api && pip3 install -r requirements.txt

# Free up some space
RUN echo "**** Cleaning up.. ****" && pip cache purge && apt-get clean

# Copy over the rest of app sources
COPY ./app /app

# Allow mounting of dependencies from docker-compose
ENV PYTHONPATH "${PYTHONPATH}:/python-dependencies"

# Add a new user and give him ownership of the application
RUN useradd -ms /bin/bash user && chown -R user:user /app

WORKDIR /app
USER user
# Default entrypoint for FastAPI
ENTRYPOINT python3 app.py
