@echo off
python grid_search_new.py BTCUSD merged_new class bagging_decisiontree --feature-selection-method importances_shap
python grid_search_new.py BTCUSD merged_new class bagging_poly_svc --feature-selection-method importances_shap
python grid_search_new.py BTCUSD merged_new class plain_knn --feature-selection-method importances_shap
python grid_search_new.py BTCUSD merged_new class plain_mnb --feature-selection-method importances_shap
python grid_search_new.py BTCUSD merged_new class plain_poly_svc --feature-selection-method importances_shap
python grid_search_new.py BTCUSD merged_new class plain_randomforest --feature-selection-method importances_shap
python grid_search_new.py BTCUSD merged_new class smote_knn --feature-selection-method importances_shap
python grid_search_new.py BTCUSD merged_new class smote_poly_svc --feature-selection-method importances_shap
python grid_search_new.py BTCUSD merged_new class plain_mlp --feature-selection-method importances_shap
python grid_search_new.py BTCUSD merged_new class smote_mlp --feature-selection-method importances_shap
python grid_search_new.py BTCUSD merged_new class plain_xgboost --feature-selection-method importances_shap
