from cryptoml.util.selection_pipeline import Pipeline
from cryptoml.util.import_proxy import SimpleImputer, StandardScaler, XGBClassifier
from joblib import cpu_count
from scipy.stats import uniform, loguniform, randint

PARAMETER_GRID = {
    'i__strategy': ['mean'],  # 'median', 'most_frequent', 'constant'
    'c__n_estimators': [100, 500],
    'c__subsample': [1], # Subsample ratio of the training instances. Setting it to 0.5 means that XGBoost would randomly sample half of the training data prior to growing trees.
    'c__colsample_bytree': [1, 0.8], # Subsample ratio of columns when constructing each tree. Subsampling occurs once for every tree constructed.
    'c__colsample_bylevel': [1, 0.8, 0.6], # Subsample ratio of columns for each level. Subsampling occurs once for every new depth level reached in a tree. Columns are subsampled from the set of columns chosen for the current tree.
    'c__colsample_bynode': [1, 0.8, 0.6],# Subsample ratio of columns for each node (split). Subsampling occurs once every time a new split is evaluated. Columns are subsampled from the set of columns chosen for the current level.
    'c__num_parallel_tree': [1], # Number of parallel trees constructed during each iteration. This option is used to support boosted random forest.
    'c__max_depth': [2, 3, 6], # Maximum depth of a tree. Increasing this value will make the model more complex and more likely to overfit.
    # 'c__reg_alpha': [0], # L1 regularization term on weights. Increasing this value will make model more conservative.
    # 'c__reg_lambda': [1], # L2 regularization term on weights. Increasing this value will make model more conservative.
    'c__learning_rate': [0.3], # 0.3 Step size shrinkage used in update to prevents overfitting. Shrinks the feature weights to make the boosting process more conservative.
    # 'c__scale_pos_weight': [1] # should be negative_samples_count / positive_samples_count
    #'c__objective': ['multi:softmax'], #XGBoost will adjust this between binary:logistic and multi:softmax based on # of classes
    'c__eval_metric': ['mlogloss'],  # logloss heavily penalizes false-positives (better precision)
    'c__tree_method': ['hist']
}

PARAMETER_DISTRIBUTION = {
    'i__strategy': ['mean'],  # 'median', 'most_frequent', 'constant'
    'c__n_estimators': randint(100, 1000),
    'c__subsample': [1], # Subsample ratio of the training instances. Setting it to 0.5 means that XGBoost would randomly sample half of the training data prior to growing trees.
    'c__colsample_bytree': uniform(0.4, 0.6), # Subsample ratio of columns when constructing each tree. Subsampling occurs once for every tree constructed.
    'c__colsample_bylevel': uniform(0.4, 0.6), # Subsample ratio of columns for each level. Subsampling occurs once for every new depth level reached in a tree. Columns are subsampled from the set of columns chosen for the current tree.
    'c__colsample_bynode': uniform(0.4, 0.6),# Subsample ratio of columns for each node (split). Subsampling occurs once every time a new split is evaluated. Columns are subsampled from the set of columns chosen for the current level.
    'c__num_parallel_tree': [1], # Number of parallel trees constructed during each iteration. This option is used to support boosted random forest.
    'c__max_depth': randint(2, 12), # Maximum depth of a tree. Increasing this value will make the model more complex and more likely to overfit.
    # 'c__reg_alpha': [0], # L1 regularization term on weights. Increasing this value will make model more conservative.
    # 'c__reg_lambda': [1], # L2 regularization term on weights. Increasing this value will make model more conservative.
    'c__learning_rate': [0.3], # 0.3 Step size shrinkage used in update to prevents overfitting. Shrinks the feature weights to make the boosting process more conservative.
    # 'c__scale_pos_weight': [1] # should be negative_samples_count / positive_samples_count
    #'c__objective': ['multi:softmax'], #XGBoost will adjust this between binary:logistic and multi:softmax based on # of classes
    'c__eval_metric': ['mlogloss'],  # logloss heavily penalizes false-positives (better precision)
    'c__tree_method': ['hist']
}

estimator = Pipeline([
    ('i', SimpleImputer()),  # Replace nan's with the median value between previous and next observation
    ('s', StandardScaler()),  # Scale data in order to center it and increase robustness against noise and outliers
    ('c', XGBClassifier(use_label_encoder=False, n_jobs=int(cpu_count()/2), verbosity=0)),
])