import typer
import yaml
from cryptoml_core.services.dataset_service import DatasetService
from cryptoml.util.shap import parse_shap_values, shap
import pandas as pd
import plotly.express as px
from matplotlib import pyplot as plt

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
    # hierarchy = load_hierarchy(f"{dataset}_{target}_feature_hierarchy.yml", importances=fs.feature_importances)

    # hdf = pd.DataFrame(hierarchy)
    # fig = px.treemap(hdf, path=['category', 'subgroup', 'name'], values='importance')
    # fig.show()
    #
    # fig = px.sunburst(hdf, path=['category', 'subgroup', 'name'], values='importance')
    # fig.show()

    shap_values, shap_expected_values = parse_shap_values(fs.shap_values)
    X = ds_service.get_dataset_features(ds=ds, begin=fs.search_interval.begin, end=fs.search_interval.end)
    y = ds_service.get_target(name='class', symbol=symbol, begin=fs.search_interval.begin, end=fs.search_interval.end)
    fig = plt.figure()
    plt.suptitle(f"Shap summary plot for {dataset}.{symbol} -> {target}")
    shap.summary_plot(shap_values, X, class_names=["SELL", "HOLD", "BUY"], show=False, max_display=352, use_log_scale=True)
    plt.tight_layout()
    fig.show()

    shap_dfs = []
    for cls, arr in enumerate(shap_values):
        class_df = pd.DataFrame(arr, columns=X.columns, index=X.index)
        class_df.columns = [f"{c}_class{cls}" for c in class_df.columns]
        shap_dfs.append(class_df)
    shap_df = pd.concat(shap_dfs, axis='columns')
    shap_df = shap_df.reindex(sorted(shap_df.columns), axis=1)
    print(shap_df.head())

    # for i in range(len(shap_expected_values)):
    #     fig = plt.figure()
    #     plt.suptitle(f"Class {i} for {dataset}.{symbol} -> {target}")
    #     shap.force_plot(
    #         base_value=shap_expected_values[i],
    #         shap_values=shap_values[i],
    #         features=X,
    #         feature_names=[c for c in X.columns],
    #         # matplotlib=True,
    #         show=False
    #     )
    #     plt.tight_layout()
    #     fig.show()

    #print("hey")


if __name__ == '__main__':
    typer.run(main)
