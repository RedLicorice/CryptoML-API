import uvicorn
from cryptoml_api.application import create_app
import typer
from typing import Optional
import logging


def main(host: Optional[str] = "0.0.0.0", port: Optional[int] = 8000):
    logging.info("<-<-[ CryptoML API ]->->\nHost: {}:{}".format(host, port))
    app = create_app()
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    typer.run(main)
