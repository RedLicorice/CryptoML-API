import base64
import io

from cryptoml.util.shap import get_shap_values
from cryptoml.util.sliding_window import test_windows, predict_day
from cryptoml.util.flattened_classification_report import flattened_classification_report_imbalanced, roc_auc_report
from cryptoml.pipelines import get_pipeline, PIPELINE_LIST
from cryptoml_core.models.dataset import FeatureSelection
import cryptoml_core.services.storage_service as storage_service
from cryptoml_core.util.timestamp import sub_interval, add_interval, from_timestamp, timestamp_windows, to_timestamp
from cryptoml_core.models.classification import Model, ModelTest, ModelParameters, ModelFeatures
from cryptoml_core.repositories.classification_repositories import ModelRepository, DocumentNotFoundException
from cryptoml_core.services.dataset_service import DatasetService
from cryptoml_core.exceptions import MessageException
import logging
import itertools
from typing import Optional
from uuid import uuid4
from cryptoml_core.util.timestamp import get_timestamp
from pydantic.error_wrappers import ValidationError
import numpy as np
import pandas as pd
from joblib import Parallel, delayed, cpu_count, dump, load
from datetime import datetime
from tqdm import tqdm


def fit_estimator_new(model: Model, mp: ModelParameters, features: str, day: str, window: dict, X, y, b, e, **kwargs):
    # Check if estimator exists
    if exist_estimator(
            model=model,
            parameters=mp.parameter_search_method,
            features=features,
            day=day,
            window=window
    ):
        logging.info(
            f"Estimator exists {model.pipeline}({model.dataset}.{model.symbol}) -> {model.target}"
            f" Day: {day} Window: {window}"
        )
        existing = load_estimator(
            model=model,
            parameters=mp.parameter_search_method,
            features=features,
            day=day,
            window=window
        )
        if existing and existing.is_fit:
            return existing
    X_train = X[:-1]
    y_train = y[:-1]

    pipeline_module = get_pipeline(model.pipeline)
    y_unique, _, y_counts = np.unique(y_train, return_index=True, return_counts=True)
    if (y_counts < 3).any():
        logging.warning(
            f"fit_estimator: y_train contains less than 3 samples for some class! \nUnique: {y_unique}\nCounts: {y_counts}")

    est = pipeline_module.estimator
    est.set_params(**mp.parameters)

    try:
        start_at = datetime.utcnow().timestamp()
        est = est.fit(X_train, y_train)
        dur = datetime.utcnow().timestamp() - start_at
    except Exception as e:
        logging.exception(f"Exception in estimator fit for day: {day}: {e}")
        return None

    # Save data as attributes of the fit estimator as well
    est.fit_time = dur
    est.fit_timestamp = get_timestamp()
    est.is_fit = True
    est.train_x = X_train
    est.train_y = y_train
    est.begin = b
    est.end = e
    est.skip_save = False
    # Training parameters and Model tuple
    est.day = day
    est.pipeline = model.pipeline
    est.dataset = model.dataset
    est.target = model.target
    est.symbol = model.symbol
    est.train_begin = to_timestamp(X_train.first_valid_index().to_pydatetime())
    est.train_end = to_timestamp(X_train.last_valid_index().to_pydatetime())
    est.window = window
    est.fit_timestamp = get_timestamp()
    est.parameters = mp.parameter_search_method
    est.features = features

    return est


def predict_estimator_day(estimator, day: str, X, y):
    if not estimator.is_fit:
        logging.exception(f"Predict window needs a fit estimator!")
    X_test = X[-1:]
    y_test = y[-1:]

    start_at = datetime.utcnow().timestamp()
    pred = estimator.predict(X_test)
    proba = estimator.predict_proba(X_test)
    predict_time = datetime.utcnow().timestamp() - start_at

    result = {
        'time': day,
        'fit_time': estimator.fit_time,
        'predict_time': predict_time,
        'predicted': pred[0],
        'label': y_test.iloc[0]
    }

    y_unique, _, y_counts = np.unique(estimator.train_y, return_index=True, return_counts=True)
    if proba.any():
        for cls, prob in enumerate(proba[0]):
            result['predicted_proba_' + str(cls)] = prob
    for u, c in zip(y_unique, y_counts):
        result[f"class_{u}_count"] = c

    return result


def shap_estimator_day(estimator, day, X, y):
    if not estimator.is_fit:
        logging.exception(f"Predict window needs a fit estimator!")
    X_test = X[-1:]
    y_test = y[-1:]

    start_at = datetime.utcnow().timestamp()
    shap_values, shap_expected = get_shap_values(estimator, X=X_test, X_train=estimator.x_train, bytes=False)
    shap_time = datetime.utcnow().timestamp() - start_at

    result = {
        'time': day,
        'shap_time': shap_time,
        'label': y_test,
        'shap_values': shap_values,
        'shap_expected': shap_expected
    }
    return result


def create_models_batch(symbol, items):
    print("Model batch: {}".format(symbol, len(items)))
    with ModelRepository() as model_repo:
        models = []
        for d, t, p in items:
            try:
                m = model_repo.find_by_symbol_dataset_target_pipeline(symbol=d.symbol, dataset=d.name, target=t,
                                                                      pipeline=p)
                logging.info("Model exists: {}-{}-{}-{}".format(d.symbol, d.name, t, p))
                models.append(m)
            except ValidationError as e:
                logging.info("Model exists and is invalid: {}-{}-{}-{}".format(d.symbol, d.name, t, p))
                pass
            except DocumentNotFoundException as e:
                m = Model(
                    symbol=d.symbol,
                    dataset=d.name,
                    target=t,
                    pipeline=p,
                )
                models.append(model_repo.create(m))
                logging.info("Model created: {}-{}-{}-{}".format(d.symbol, d.name, t, p))
                pass
        return models


def get_estimator_day_name(model: Model, day: str, window: dict, parameters: str, features: str):
    window_str = ''
    for k, v in window.items():
        window_str += f'{k}-{v},'
    window_str = window_str[:-1]
    day_clean = day.replace('-', '').replace('_', '').replace(':', '').replace('+0000', '')
    result = f'{model.dataset}/{model.pipeline}/{model.symbol}/{model.target}__{parameters}__{features}__{day_clean}_W{window_str}.p'
    return result


def save_estimator(estimator):
    if not estimator:
        return False

    filename = get_estimator_day_name(
        model=estimator,
        day=estimator.day,
        window=estimator.window,
        parameters=estimator.parameters,
        features=estimator.features
    )
    if estimator.skip_save:
        logging.info(
            f"Skip save {estimator.pipeline}({estimator.dataset}.{estimator.symbol}, D={estimator.day}, W={estimator.window}) -> {estimator.target}"
        )
        return filename

    output = io.BytesIO()
    dump_res = dump(value=estimator, filename=output, compress=5)
    estimator.skip_save = True
    data = output.getvalue()
    return storage_service.upload_pickle_obj(data, bucket='fit-estimators', name=filename)


def load_estimator(model: Model, day: str, window: dict, parameters: str, features: str):
    filename = get_estimator_day_name(model=model, day=day, window=window, parameters=parameters,
                                      features=features)
    # if not storage_service.exist_file(bucket='fit-estimators', name=filename):
    #     return None

    data_file = storage_service.load_pickled_obj(bucket='fit-estimators', name=filename)
    if not data_file:
        return None
    input = io.BytesIO(data_file)
    est = load(input)
    est.skip_save = True
    return est


def exist_estimator(model: Model, day: str, window: dict, parameters: str, features: str):
    filename = get_estimator_day_name(model=model, day=day, window=window, parameters=parameters,
                                                   features=features)
    return storage_service.exist_file(bucket='fit-estimators', name=filename)


class ModelService:
    def __init__(self):
        self.model_repo: ModelRepository = ModelRepository()
        self.dataset_service = DatasetService()

    def create_classification_models(self, query, pipeline):
        ds = DatasetService()
        models = []
        if query is None:
            query = {
                {"type": "FEATURES", }
            }
        datasets = ds.query(query)
        # All possible combinations
        all_models = {}
        for d in datasets:
            # Get targets for this symbol
            tgt = ds.get_dataset('target', d.symbol)
            if not d.symbol in all_models:
                all_models[d.symbol] = []
            for t, p in itertools.product(tgt.features, PIPELINE_LIST):
                if t in ['price', 'pct']:
                    continue
                all_models[d.symbol].append((d, t, p))
        # Method to process a batch of items
        results = Parallel(n_jobs=-1)(
            delayed(create_models_batch)(symbol, items) for symbol, items in all_models.items())
        return [item for sublist in results for item in sublist]

    def clear_features(self, query=None):
        return self.model_repo.clear_features(query or {})

    def clear_parameters(self, query=None):
        return self.model_repo.clear_parameters(query or {})

    def clear_tests(self, query=None):
        return self.model_repo.clear_tests(query or {})

    def all(self):
        return [m for m in self.model_repo.iterable()]

    @staticmethod
    def get_model_parameters(m: Model, method: str):
        for mp in m.parameters:
            if mp.parameter_search_method == method:
                return mp
        return None

    def remove_parameters(self, model: Model, method: str):
        found = None
        for i in range(len(model.parameters)):
            if model.parameters[i].parameter_search_method == method:
                found = i
        if found is not None:
            del model.parameters[found]
            self.model_repo.update(model.id, model)
            return True
        return False

    def get_model(self, model_id):
        return self.model_repo.get(model_id)

    def get_model(self, pipeline: str, dataset: str, target: str, symbol: str):
        result = self.model_repo.query({"symbol": symbol, "dataset": dataset, "target": target, "pipeline": pipeline})
        if not result:
            return None
        return result[0]

    def get_test(self, pipeline: str, dataset: str, target: str, symbol: str, window: int):
        # result = self.model_repo.get_model_test(pipeline, dataset, target, symbol, window)
        # if not result:
        #     return None
        # return result[0]
        model = self.get_model(pipeline=pipeline, dataset=dataset, target=target, symbol=symbol)
        for t in model.tests:
            if t.window['days'] == window:
                return t
        return None

    @staticmethod
    def parse_test_results(test: ModelTest):
        if isinstance(test, dict):
            test = ModelTest(**test)
        # Re-convert classification results from test to a DataFrame
        results = pd.DataFrame(test.classification_results)
        # Parse index so it's a DateTimeIndex, because Mongo stores it as a string
        results.index = pd.to_datetime(results.time)
        return results

    def get_test_results(self, pipeline: str, dataset: str, target: str, symbol: str, window: int):
        test = self.get_test(pipeline, dataset, target, symbol, window)
        return ModelService.parse_test_results(test)

    def query_models(self, query, projection: Optional[dict] = None):
        return self.model_repo.query(query, projection)

    def create_model_test(self, *, model: Model, split=0.7, step=None, task_key=None, window=None, **kwargs):
        service = DatasetService()
        ds = service.get_dataset(model.dataset, model.symbol)
        splits = DatasetService.get_train_test_split_indices(ds, split)
        parameters = kwargs.get('parameters')
        features = kwargs.get('features')
        if isinstance(parameters, str) and parameters == 'latest':
            if model.parameters:
                parameters = model.parameters[-1].parameters
            else:
                parameters = None

        if isinstance(features, str):
            fs = DatasetService.get_feature_selection(ds=ds, method=features, target=model.target)
            if fs:
                features = fs.features
            else:
                features = None
        result = ModelTest(
            window=window or {'days': 30},
            step=step or ds.interval,
            parameters=parameters or {},
            features=features or [],
            test_interval=splits['test'],
            task_key=task_key or str(uuid4())
        )
        return result

    def test_model(self, model: Model, mt: ModelTest, **kwargs):
        if not model.id:
            model = self.model_repo.create(model)
        if self.model_repo.exist_test(model.id, mt.task_key):
            logging.info("Model {} test {} already executed!".format(model.id, mt.task_key))
            return mt
        # Load dataset
        ds = DatasetService()
        d = ds.get_dataset(model.dataset, model.symbol)
        # Get training data including the first training window
        begin = sub_interval(timestamp=mt.test_interval.begin, interval=mt.window)
        end = add_interval(timestamp=mt.test_interval.end, interval=mt.step)
        if from_timestamp(d.valid_index_min).timestamp() > from_timestamp(begin).timestamp():
            raise MessageException("Not enough data for training! [Pipeline: {} Dataset: {} Symbol: {} Window: {}]" \
                                   .format(model.pipeline, model.dataset, model.symbol, mt.window))
        X = ds.get_features(model.dataset, model.symbol, begin=begin, end=end)
        y = ds.get_target(model.target, model.symbol, begin=begin, end=end)

        unique, counts = np.unique(y, return_counts=True)
        if len(unique) < 2:
            logging.error("[{}-{}-{}-{}]Training data contains less than 2 classes: {}"
                          .format(model.symbol, model.dataset, model.target, model.pipeline, unique))
            raise MessageException("Training data contains less than 2 classes: {}".format(unique))

        # Load pipeline
        pipeline_module = get_pipeline(model.pipeline)
        # Slice testing interval in windows

        ranges = timestamp_windows(begin, end, mt.window, mt.step)

        mt.start_at = get_timestamp()
        df = test_windows(pipeline_module.estimator, mt.parameters, X, y, ranges)
        mt.end_at = get_timestamp()

        mt.classification_results = df.to_dict()

        clf_report = flattened_classification_report_imbalanced(df.label, df.predicted)
        roc_report = roc_auc_report(df.label, df.predicted, df[[c for c in df.columns if '_proba_' in c]])
        clf_report.update(roc_report)
        mt.classification_report = clf_report

        self.model_repo.append_test(model.id, mt)

        return mt

    def test_model_new(self, *, pipeline: str, dataset: str, symbol: str, target: str, split=0.7, step=None,
                       task_key=None, window=None, **kwargs):
        test_window = window or {'days': 90}
        model = self.get_model(pipeline=pipeline, dataset=dataset, symbol=symbol, target=target)
        # for t in enumerate(model.tests):
        #     if t['window']['days'] == test_window['days']:
        #         if not kwargs.get('force'):
        #             logging.info(f"Model {pipeline}({dataset}.{symbol}) -> {target} "
        #                          f"test with window {test_window} already executed!")
        #             if kwargs.get('save'):
        #                 return t

        ds = self.dataset_service.get_dataset(dataset, symbol)
        splits = DatasetService.get_train_test_split_indices(ds, split)
        test_interval = splits['test']
        test_step = step or ds.interval

        # Parse model parameters: if it's a string, give it an interpretation
        parameters = kwargs.get('parameters')
        features = kwargs.get('features')
        mp = ModelService.get_model_parameters(m=model, method=parameters)
        if not mp:
            logging.warning(f"Parameter search with method {parameters} does not exist in model"
                            f" {model.pipeline}({model.dataset}.{model.symbol}) -> {model.target}")

        # Get training data including the first training window
        begin = sub_interval(timestamp=test_interval["begin"], interval=test_window)
        end = add_interval(timestamp=test_interval["end"], interval=test_step)
        if from_timestamp(ds.valid_index_min).timestamp() > from_timestamp(begin).timestamp():
            raise MessageException(f"Not enough data for training with window {test_window}!"
                                   f" {model.pipeline}({model.dataset}.{model.symbol}) -> {model.target}")
        test_X, test_y = self.dataset_service.get_x_y(dataset, symbol, target, features, begin, end)
        # Slice testing interval in "sliding" windows
        windows = [(b, e) for b, e in timestamp_windows(begin, end, test_window, test_step)]

        # Fit the models and make predictions
        storage_service.create_bucket(bucket='fit-estimators')

        _n_jobs = int(kwargs.get('n_jobs', cpu_count() / 2))
        logging.info(f"Fitting {len(windows)} estimators with {_n_jobs} threads..")
        fit_estimators = Parallel(n_jobs=_n_jobs)(
            delayed(fit_estimator_new)(
                model=model,
                mp=mp,
                features=features,
                day=e,
                window=test_window,
                X=test_X[b:e],
                y=test_y[b:e],
                b=b,
                e=e,
                force=not kwargs.get('save')
            ) for b, e in tqdm(windows))

        logging.info(f"Saving {len(windows)} fit estimators with {_n_jobs} threads..")
        estimator_names = Parallel(n_jobs=_n_jobs)(
            delayed(save_estimator)(
                estimator=est,
            )
            for est in tqdm(fit_estimators))

        # logging.info(f"Loading {len(windows)} estimators with {_n_jobs} threads..")
        # load_estimators = Parallel(n_jobs=_n_jobs)(
        #     delayed(load_estimator)(
        #         model=model,
        #         day=e,
        #         window=window,
        #         parameters=parameters,
        #         features=features
        #     )
        #     for b, e in tqdm(windows))

        logging.info(f"Predicing {len(windows)} estimators with {_n_jobs} threads..")
        prediction_results = Parallel(n_jobs=_n_jobs)(
            delayed(predict_estimator_day)(
                estimator=est,
                day=est.day,
                X=test_X[est.begin: est.end],
                y=test_y[est.begin: est.end]
            )
            for est in tqdm(fit_estimators))

        results = [r for r in prediction_results if r is not None]
        df = pd.DataFrame(results)
        if df.empty:
            raise MessageException("TestWindows: Empty result dataframe!")
        #df.time = pd.to_datetime(df.time)
        #df = df.set_index('time')

        classification_records = [r for r in df.to_dict(orient='records')]
        # If save is true, save test instance and parameters
        mt = ModelTest(
            window=test_window,
            step=test_step,
            parameters=mp.parameters,
            features=[c for c in test_X.columns],
            test_interval=splits['test'],
            task_key=task_key or str(uuid4()),
            classification_results=classification_records,
        )
        # Populate classification report fields
        clf_report = flattened_classification_report_imbalanced(df.label, df.predicted)
        roc_report = roc_auc_report(df.label, df.predicted, df[[c for c in df.columns if '_proba_' in c]])
        clf_report.update(roc_report)
        mt.classification_report = clf_report

        # Save test into the model
        if kwargs.get('save'):
            return self.model_repo.append_test(model.id, mt)
        return mt

    # def get_test_models(self, *, pipeline: str, dataset: str, symbol: str, target: str, split=0.7, step=None,
    #                    task_key=None, window=None, **kwargs):
    #     _n_jobs = int(kwargs.get('n_jobs', cpu_count() / 2))
    #     model = self.get_model(pipeline=pipeline, dataset=dataset, symbol=symbol, target=target)
    #     ds = self.dataset_service.get_dataset(name=model.dataset, symbol=model.symbol)
    #     for t in enumerate(model.tests):
    #         self.dataset_service
    #         estimator_names = Parallel(n_jobs=_n_jobs)(
    #             delayed(load_estimator)(
    #                 estimator=est,
    #                 model=model,
    #                 parameters=t.parameter_search_method,
    #                 features=features,
    #                 day=day,
    #                 window=window
    #             )
    #             for est in tqdm(fit_estimators))

    @staticmethod
    def load_test_estimators(model: Model, mt: ModelTest, **kwargs):
        results = ModelService.parse_test_results(mt)
        test_days = [d for d in results.time]
        _n_jobs = int(kwargs.get('n_jobs', cpu_count() / 2))
        logging.info(f"Loading {len(test_days)} estimators..")
        estimators = Parallel(n_jobs=_n_jobs)(
            delayed(load_estimator)(
                model=model,
                parameters='gridsearch',
                features='importances_shap',
                day=day,
                window=mt.window
            )
            for day in tqdm(test_days))

        return estimators

    def compare_models(self, symbol: str, dataset: str, target: str, pipeline: Optional[str] = None):
        if pipeline:
            tests = self.model_repo.find_tests(symbol=symbol, dataset=dataset, target=target, pipeline=pipeline)
        else:
            tests = self.model_repo.find_tests(symbol=symbol, dataset=dataset, target=target)
        return tests

    def predict_day(self, pipeline: str, dataset: str, target: str, symbol: str, day: str, window: dict):
        model = self.get_model(pipeline, dataset, target, symbol)
        # Load dataset
        ds = DatasetService()
        d = ds.get_dataset(model.dataset, model.symbol)
        # Get training data including the first training window
        begin = sub_interval(timestamp=day, interval=window)
        if from_timestamp(d.valid_index_min).timestamp() > from_timestamp(begin).timestamp():
            raise MessageException("Not enough data for training! [Pipeline: {} Dataset: {} Symbol: {} Window: {}]" \
                                   .format(model.pipeline, model.dataset, model.symbol, window))
        X = ds.get_features(model.dataset, model.symbol, begin=begin, end=day)
        y = ds.get_target(model.target, model.symbol, begin=begin, end=day)

        unique, counts = np.unique(y, return_counts=True)
        if len(unique) < 2:
            logging.error("[{}-{}-{}-{}]Training data contains less than 2 classes: {}"
                          .format(model.symbol, model.dataset, model.target, model.pipeline, unique))
            raise MessageException("Training data contains less than 2 classes: {}".format(unique))

        # Load pipeline
        pipeline_module = get_pipeline(model.pipeline)
        # Slice testing interval in windows

        df = predict_day(pipeline_module.estimator, model.parameters[-1], X, y, day)

        return df
