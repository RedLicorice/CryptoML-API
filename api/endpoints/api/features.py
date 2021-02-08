from fastapi import APIRouter, Form, File, UploadFile, Depends
from werkzeug.utils import secure_filename
import traceback as tb
from ...services import StorageService

router = APIRouter()


@router.post('/upload')
def upload(
        symbol: str = Form(...),
        dataset: str = Form(...),
        file: UploadFile = File(...),
        storage: StorageService = Depends(StorageService)
    ):
    filename = secure_filename(file.filename)
    #upload_file.save(filename)
    try:
        storage.upload_fileobj(file.file, symbol, "{}.{}".format(dataset, filename))
    except Exception as e:
        return {'message': 'Upload Failed', 'traceback': '\n'.join(tb.format_exception(None, e, e.__traceback__))}
    else:
        return {'message': 'Upload OK'}