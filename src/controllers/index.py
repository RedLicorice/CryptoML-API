from flask import request, jsonify
from ._routes import endpoint, url_routes

@endpoint('/', 'index')
def index():
    # Return an endpoint:rule list
    result = { u['endpoint']: u['rule'] for u in url_routes }
    return jsonify(result)