import typer
from cryptoml_core.services.model_service import ModelService
from cryptoml_core.util.timestamp import get_timestamp
from cryptoml_core.exceptions import MessageException
import json
from typing import Optional
from cryptoml_core.logging import setup_file_logger
import logging


def main(queryfile: str, features: Optional[str] = None, parameters: Optional[str] = None):
    models = ModelService()
    with open(queryfile, 'r') as f:
        query = json.load(f)
    models.clear_tests(query)
    test_models = models.query_models(query)
    logging.info("[i] {} models to test".format(len(test_models)))
    failed = []
    for i, m in enumerate(test_models):
        logging.info("==[{}/{}]== MODEL: {} {} {} {} =====".format(i + 1, len(test_models), m.symbol, m.dataset, m.target,
                                                            m.pipeline))
        t1 = models.create_model_test(model=m, split=0.7, step={'days': 1}, window={'days': 30}, parameters=parameters, features=features)
        t2 = models.create_model_test(model=m, split=0.7, step={'days': 1}, window={'days': 90}, parameters=parameters, features=features)
        t3 = models.create_model_test(model=m, split=0.7, step={'days': 1}, window={'days': 150}, parameters=parameters, features=features)
        try:
            # Test T1
            logging.info("[{}] {} Start T1".format(get_timestamp(), m.symbol))
            models.test_model(m, t1, sync=True)
            # Test T2
            logging.info("[{}] {} Start T2".format(get_timestamp(), m.symbol))
            models.test_model(m, t2, sync=True)
            # Test T3
            logging.info("[{}] {} Start T3".format(get_timestamp(), m.symbol))
            models.test_model(m, t3, sync=True)
        except MessageException as e:
            logging.error("[!] " + e.message)
            failed.append((m.dict(), t1.dict(), t2.dict(), t3.dict()))
            pass
        except Exception as e:
            logging.exception("[!] " + str(e))
            failed.append((m.dict(), t1.dict(), t2.dict(), t3.dict()))
            pass

        logging.info("[{}] Done".format(m.symbol))
    with open('test-failed.json', 'w') as f:
        json.dump(failed, f)


if __name__ == '__main__':
    setup_file_logger('test_model.log')
    typer.run(main)
