import typer
from cryptoml_core.services.model_service import ModelService
from cryptoml_core.util.timestamp import get_timestamp
from cryptoml_core.exceptions import MessageException
import json
from typing import Optional


def main(dataset: str, target: str, pipeline: str, features: Optional[str] = None, parameters: Optional[str] = None):
    models = ModelService()
    query = {"dataset": dataset, "target": target, "pipeline": pipeline}
    if pipeline == 'all':
        del query['pipeline']
    if target == 'all':
        del query['target']
    models.clear_tests(query)
    test_models = models.query_models(query)
    print("[i] {} models to test".format(len(test_models)))
    failed = []
    for i, m in enumerate(test_models):
        print("==[{}/{}]== MODEL: {} {} {} {} =====".format(i + 1, len(test_models), m.symbol, m.dataset, m.target,
                                                            m.pipeline))
        t1 = models.create_model_test(model=m, split=0.7, step={'days': 1}, window={'days': 30}, parameters=parameters, features=features)
        t2 = models.create_model_test(model=m, split=0.7, step={'days': 1}, window={'days': 90}, parameters=parameters, features=features)
        t3 = models.create_model_test(model=m, split=0.7, step={'days': 1}, window={'days': 150}, parameters=parameters, features=features)
        try:
            # Test T1
            print("[{}] {} Start T1".format(get_timestamp(), m.symbol))
            models.test_model(m, t1, sync=True)
            # Test T2
            print("[{}] {} Start T2".format(get_timestamp(), m.symbol))
            models.test_model(m, t2, sync=True)
            # Test T3
            print("[{}] {} Start T3".format(get_timestamp(), m.symbol))
            models.test_model(m, t3, sync=True)
        except MessageException as e:
            print("[!] " + e.message)
            failed.append((m, t1, t2, t3))
            pass
        except Exception as e:
            print("[!] " + str(e))
            failed.append((m, t1, t2, t3))
            pass

        print("[{}] Done".format(m.symbol))
    with open('test-failed.json', 'w') as f:
        json.dump(failed, f)


if __name__ == '__main__':
    typer.run(main)