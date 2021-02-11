from fastapi import APIRouter, Form, File, UploadFile, Depends
from werkzeug.utils import secure_filename
import traceback as tb
from ...services import StorageService, FeatureService

router = APIRouter()


@router.post('/upload')
def upload(
        symbol: str = Form(...),
        dataset: str = Form(...),
        file: UploadFile = File(...),
        storage: StorageService = Depends(StorageService),
        features: FeatureService = Depends(FeatureService)
    ):
    # CSV mime-type is "text/csv"
    if file.content_type != 'text/csv':
        return {'message': 'Upload failed: wrong content type!'}
    # Store uploaded dataset on S3
    filename = secure_filename(file.filename)
    try:
        storage.upload_fileobj(file.file, 'temp-uploads', "{}.{}".format(dataset, filename))
    except Exception as e:
        return {'message': 'Failed to save CSV on s3 storage', 'traceback': '\n'.join(tb.format_exception(None, e, e.__traceback__))}
    # Import dataset to feature repository
    try:
        features.import_from_storage('temp-uploads', "{}.{}".format(dataset, filename), dataset, symbol)
    except Exception as e:
        return {'message': 'Failed to import CSV in feature repository', 'traceback': '\n'.join(tb.format_exception(None, e, e.__traceback__))}
    # Return dataset spec
    return {'message': 'Upload OK', 'content_type': file.content_type, 'symbol':symbol, 'dataset':dataset}