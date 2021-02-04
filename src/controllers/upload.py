from flask import request, jsonify
from werkzeug.utils import secure_filename
from dependency_injector.wiring import inject, Provide
from ..services import FeaturesService
from ..containers import Container
from ._routes import add_endpoint
import traceback as tb
import boto3

@inject
def upload(
        feature_service: FeaturesService = Provide[Container.feature_service],
        storage = Provide[Container.storage_service]
    ):
    upload_file = request.files['file']
    symbol = request.form['symbol']
    dataset = request.form['dataset']

    if upload_file:
        filename = secure_filename(upload_file.filename)
        #upload_file.save(filename)
        try:
            storage.upload_fileobj(upload_file, symbol, "{}.{}".format(dataset, filename))
        except Exception as e:
            result = {'message': 'Upload Failed', 'traceback': '\n'.join(tb.format_exception(None, e, e.__traceback__))}
            return jsonify(result)
        else:
            result = {'message': 'Upload OK'}
            return jsonify(result)

# Dependency-injector does not like multiple decorators attached!
add_endpoint('/upload', 'upload', upload, methods=['POST'])