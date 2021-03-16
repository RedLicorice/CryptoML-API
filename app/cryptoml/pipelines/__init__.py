import importlib
import logging

PIPELINE_LIST = [
    # 'adaboost_decisiontree',
    'bagging_decisiontree',
    # 'bagging_linear_svc',
    'bagging_poly_svc',
    # 'bagging_rbf_svc',
    # 'debug_xgboost',
    'plain_knn',
    # 'plain_linear_svc',
    'plain_mlp',
    'plain_mnb',
    'plain_poly_svc',
    'plain_randomforest',
    # 'plain_rbf_svc',
    'plain_xgboost',
    'smote_knn',
    'smote_mlp',
    'smote_poly_svc'
]

# Dinamically import and validate a pipeline from cryptoml.pipelines.*
def get_pipeline(pipeline, unlisted=False):
    if not pipeline in PIPELINE_LIST and not unlisted:
        raise Exception('Package cryptoml.pipelines has no {} module!'.format(pipeline))
    try:
        pipeline_module = importlib.import_module('cryptoml.pipelines.{}'.format(pipeline))
        if not pipeline_module:
            raise Exception('Failed to import cryptoml.pipelines.{} (importlib returned None)!'.format(pipeline))
        if not hasattr(pipeline_module, 'estimator'):
            raise Exception('Builder cryptoml.pipelines.{} has no "estimator" attribute!'.format(pipeline))
    except Exception as e:
        logging.exception(e)
        raise Exception('Failed to import cryptoml.pipelines.{} !'.format(pipeline))
    return pipeline_module