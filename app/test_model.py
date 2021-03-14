import typer
from cryptoml_core.services.model_service import ModelService
from cryptoml_core.util.timestamp import get_timestamp
from cryptoml_core.exceptions import MessageException


def main(dataset, target, pipeline):
    models = ModelService()
    query = {"dataset": dataset, "target": target, "pipeline": pipeline}
    models.clear_tests(query)
    test_models = models.query_models(query)
    print("[i] {} models to test".format(len(test_models)))
    for i, m in enumerate(test_models):
        print("==[{}/{}]== MODEL: {} {} {} {} =====".format(i + 1, len(test_models), m.symbol, m.dataset, m.target,
                                                            m.pipeline))
        t1 = models.create_model_test(model=m, split=0.7, step={'days': 1}, window={'days': 30})
        t2 = models.create_model_test(model=m, split=0.7, step={'days': 1}, window={'days': 90})
        t3 = models.create_model_test(model=m, split=0.7, step={'days': 1}, window={'days': 150})
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
            pass

        print("[{}] Done".format(m.symbol))


if __name__ == '__main__':
    typer.run(main)
