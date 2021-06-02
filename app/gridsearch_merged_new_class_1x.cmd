@echo off
python grid_search_new.py ADAUSD merged_new class plain_xgboost --feature-selection-method importances_shap
python grid_search_new.py BCHUSD merged_new class plain_xgboost --feature-selection-method importances_shap
python grid_search_new.py BNBUSD merged_new class plain_xgboost --feature-selection-method importances_shap
python grid_search_new.py BTGUSD merged_new class plain_xgboost --feature-selection-method importances_shap
python grid_search_new.py DASHUSD merged_new class plain_xgboost --feature-selection-method importances_shap
python grid_search_new.py DOGEUSD merged_new class plain_xgboost --feature-selection-method importances_shap