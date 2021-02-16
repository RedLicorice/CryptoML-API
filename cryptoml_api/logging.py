import logging
import sys
# ToDo: Use https://github.com/paksu/pytelegraf to ingest logs into influxdb
# https://geekflare.com/open-source-centralized-logging/

def setup_logger():
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(levelname)s - %(asctime)s - %(name)s: %(message)s", "%Y-%m-%d %H:%M:%S")
    handler.setFormatter(formatter)
    root.addHandler(handler)