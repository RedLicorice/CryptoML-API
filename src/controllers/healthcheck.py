from flask import jsonify
from datetime import datetime
from ._routes import endpoint

@endpoint('/health-check', 'health-check')
def healthcheck():
    return jsonify(datetime.now().strftime("%b %d %Y %H:%M:%S"))