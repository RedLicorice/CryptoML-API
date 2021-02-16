import uvicorn
from cryptoml_api.application import create_app
import logging, sys


app = create_app()

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)