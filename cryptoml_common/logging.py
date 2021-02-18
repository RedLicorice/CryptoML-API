import logging
import sys

# Setup logging to STDOUT so we can use a common log collector for centralizing logs in the future.
def setup_logger():
    root = logging.getLogger()
    root.setLevel(logging.INFO)

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(levelname)s - %(asctime)s - %(name)s: %(message)s", "%Y-%m-%d %H:%M:%S")
    handler.setFormatter(formatter)
    root.addHandler(handler)