import typer
import yaml
from cryptoml_core.services.dataset_service import DatasetService
from cryptoml.util.shap import parse_shap_values, shap
import pandas as pd
import os
import numpy as np

# Load hierarchy in normalized form (dataframe) with feature importances
def load_hierarchy(filename: str, importances: dict, window=10):
    with open(filename, 'r') as f:
        data = yaml.load(f)
    base = data['features']

    # expanded = []
    # for f in base['ohlcv']['history']:
    #     for i in range(1, window+1, 1):
    #         expanded.append(f.format(i))
    # base['ohlcv']['history'] = expanded
    def get_importance(name: str, importances: dict):
        try:
            return importances[name]
        except KeyError as e:
            print(f"Feature {name} not in feature_importances!")
            return 0.00

    result = []
    for category, group in base.items():
        for subgroup, features in group.items():
            for feature in features:
                if '{}' in feature:
                    for i in range(1, window + 1, 1):
                        f_name = feature.format(i)
                        result.append({
                            'category': category,
                            'subgroup': subgroup,
                            'name': f_name,
                            'importance': get_importance(f_name, importances)
                        })
                else:
                    result.append({
                        'category': category,
                        'subgroup': subgroup,
                        'name': feature,
                        'importance': get_importance(feature, importances)
                    })
    return result


def main(dataset: str, target: str, symbol: str):
    ds_service = DatasetService()
    ds = ds_service.get_dataset(name=dataset, symbol=symbol)
    fs = DatasetService.get_feature_selection(ds=ds, method='importances_shap', target='class')

    # shap_values, shap_expected_values = parse_shap_values(fs.shap_values)
    # X = ds_service.get_dataset_features(ds=ds, begin=fs.search_interval.begin, end=fs.search_interval.end)

    # shap_df_0 = pd.DataFrame(data=shap_values[0], index=X.index, columns=X.columns)
    # shap_df_1 = pd.DataFrame(data=shap_values[1], index=X.index, columns=X.columns)
    # shap_df_2 = pd.DataFrame(data=shap_values[2], index=X.index, columns=X.columns)

    hierarchy = load_hierarchy(f"{dataset}_{target}_feature_hierarchy.yml", importances=fs.feature_importances)
    # for record in hierarchy:
    #     feature = record['name']
    #     try:
    #         record['shap_mean_0'] = shap_df_0[feature].mean()
    #         record['shap_mean_1'] = shap_df_1[feature].mean()
    #         record['shap_mean_2'] = shap_df_2[feature].mean()
    #     except KeyError as e:
    #         print(f"Feature {feature} not in dataset!")
    #         record['shap_mean_0'] = np.nan
    #         record['shap_mean_1'] = np.nan
    #         record['shap_mean_2'] = np.nan
    #         pass


    os.makedirs(f"data/selection_{dataset}_{target}/", exist_ok=True)

    hdf = pd.DataFrame(hierarchy)
    csv_name = f"data/selection_{dataset}_{target}/{symbol}_feature_importances.csv"
    hdf.to_csv(csv_name, index_label='index')
    print(f"Augmented importances dataframe exported to {csv_name}")

    csv_name = f"data/selection_{dataset}_{target}/{symbol}_feature_importances_selected.csv"
    hdf[hdf.name.isin(fs.features)].to_csv(csv_name, index_label='index')
    print(f"Augmented selected features dataframe exported to {csv_name}")

    # fig = px.sunburst(hdf, path=['category', 'subgroup', 'name'], values='importance', title=f"{symbol} XGBoost feature importances")
    # fig.show()
    #
    # filtered_hdf = hdf.dropna(axis='rows', how='any')
    # fig = px.sunburst(filtered_hdf, path=['category', 'subgroup', 'name'], values='shap_mean_0', title=f"{symbol} mean SHAP values for class SELL")
    # fig.show()
    # fig = px.sunburst(filtered_hdf, path=['category', 'subgroup', 'name'], values='shap_mean_1', title=f"{symbol} mean SHAP values for class HOLD")
    # fig.show()
    # fig = px.sunburst(filtered_hdf, path=['category', 'subgroup', 'name'], values='shap_mean_2', title=f"{symbol} mean SHAP values for class BUY")
    # fig.show()



if __name__ == '__main__':
    typer.run(main)
