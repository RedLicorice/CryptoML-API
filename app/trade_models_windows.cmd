@echo off
set symbol=%1
python trading_simulation.py bagging_decisiontree merged_new %symbol% 90
python trading_simulation.py bagging_poly_svc merged_new %symbol% 90
python trading_simulation.py plain_knn merged_new %symbol% 90
python trading_simulation.py plain_mnb merged_new %symbol% 90
python trading_simulation.py plain_poly_svc merged_new %symbol% 90
python trading_simulation.py plain_randomforest merged_new %symbol% 90
python trading_simulation.py smote_knn merged_new %symbol% 90
python trading_simulation.py smote_poly_svc merged_new %symbol% 90
python trading_simulation.py plain_mlp merged_new %symbol% 90
python trading_simulation.py smote_mlp merged_new %symbol% 90
python trading_simulation.py plain_xgboost merged_new %symbol% 90
python trading_simulation.py bagging_decisiontree merged_new %symbol% 180
python trading_simulation.py bagging_poly_svc merged_new %symbol% 180
python trading_simulation.py plain_knn merged_new %symbol% 180
python trading_simulation.py plain_mnb merged_new %symbol% 180
python trading_simulation.py plain_poly_svc merged_new %symbol% 180
python trading_simulation.py plain_randomforest merged_new %symbol% 180
python trading_simulation.py smote_knn merged_new %symbol% 180
python trading_simulation.py smote_poly_svc merged_new %symbol% 180
python trading_simulation.py plain_mlp merged_new %symbol% 180
python trading_simulation.py smote_mlp merged_new %symbol% 180
python trading_simulation.py plain_xgboost merged_new %symbol% 180
python trading_simulation.py bagging_decisiontree merged_new %symbol% 240
python trading_simulation.py bagging_poly_svc merged_new %symbol% 240
python trading_simulation.py plain_knn merged_new %symbol% 240
python trading_simulation.py plain_mnb merged_new %symbol% 240
python trading_simulation.py plain_poly_svc merged_new %symbol% 240
python trading_simulation.py plain_randomforest merged_new %symbol% 240
python trading_simulation.py smote_knn merged_new %symbol% 240
python trading_simulation.py smote_poly_svc merged_new %symbol% 240
python trading_simulation.py plain_mlp merged_new %symbol% 240
python trading_simulation.py smote_mlp merged_new %symbol% 240
python trading_simulation.py plain_xgboost merged_new %symbol% 240
