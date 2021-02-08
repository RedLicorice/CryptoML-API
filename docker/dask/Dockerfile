FROM python:3.7
RUN apt-get update -y && apt-get install -y python3-pip libsnappy-dev
RUN pip install numpy
RUN pip install dask
RUN pip install distributed
RUN pip install fsspec
RUN pip install fastavro
RUN pip install python-snappy
RUN pip install dask[bag]
RUN pip install dask[dataframe]
RUN pip install jupyter-server-proxy
RUN pip install bokeh>=0.13.0

# Dashboard
EXPOSE 8787
# Scheduler
EXPOSE 8786
