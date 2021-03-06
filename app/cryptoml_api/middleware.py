from fastapi import Request, HTTPException
from cryptoml_core.exceptions import MessageException, NotFoundException
import logging

__all__ = ("request_handler",)


async def request_handler(request: Request, call_next):
    """Middleware used to process each request on FastAPI, to provide error handling (convert exceptions to responses).
    TODO: add logging and individual request traceability
    """
    try:
        return await call_next(request)

    except NotFoundException as ex:
        raise HTTPException(status_code=404, detail=ex.message)

    except Exception as ex:
        logging.exception(ex)
        if isinstance(ex, MessageException):
            return ex.message
        # Re-raising other exceptions will return internal error 500 to the client
        raise ex