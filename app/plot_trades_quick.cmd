@echo off
set symbol=%1
python trading_plotly.py bagging_decisiontree merged_new
python trading_plotly.py bagging_poly_svc merged_new
python trading_plotly.py plain_knn merged_new
python trading_plotly.py plain_mnb merged_new
python trading_plotly.py plain_poly_svc merged_new
python trading_plotly.py plain_randomforest merged_new
python trading_plotly.py smote_knn merged_new
python trading_plotly.py smote_poly_svc merged_new
python trading_plotly.py plain_mlp merged_new
python trading_plotly.py smote_mlp merged_new
python trading_plotly.py plain_xgboost merged_new
