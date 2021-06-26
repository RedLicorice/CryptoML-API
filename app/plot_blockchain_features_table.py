import typer
import yaml
from cryptoml_core.services.dataset_service import DatasetService
from cryptoml.util.shap import parse_shap_values, shap
import pandas as pd
import plotly.graph_objects as go
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


def main(dataset: str):
    ds_service = DatasetService()
    symbols = ds_service.get_dataset_symbols(name=dataset)
    ds_data = {s: ds_service.get_dataset(name=dataset, symbol=s).features for s in symbols}
    # We need to reshape / flatten data
    records = []
    symbol_lookup = {s:i for i, s in enumerate(symbols)}
    for symbol, features in ds_data.items():
        record = {
            'symbol': symbol.replace('USD', ''),
            #'symbol_id': symbol_lookup[symbol]
        }
        for f in features:
            if f.startswith('adrbal1in') and f.endswith('cnt'):
                f = 'adrbal1in{N}cnt'
            elif f.startswith('adrbalntv') and f.endswith('cnt'):
                f = 'adrbalntv{N}cnt'
            elif f.startswith('splyact') and not 'pct' in f:
                f = 'splyact{T}'
            elif f.startswith('splyadrbal1in'):
                f = 'splyadrbal1in{N}'
            elif f.startswith('splyadrbalntv'):
                f = 'splyadrbalntv{N}'
            elif f.startswith('splyadrtop'):
                f = 'splyadrtop{N}'
            elif f.startswith('adrbalusd') and f.endswith('cnt'):
                f = 'adrbalusd{N}cnt'
            elif f.startswith('splyadrbalusd'):
                f = 'splyadrbalusd{N}'
            elif f.startswith('txtfrval') and f.endswith('ntv'):
                f = 'txtfrval{A}ntv'
            elif f.startswith('txtfrval') and f.endswith('usd'):
                f = 'txtfrval{A}usd'
            elif f.startswith('fee') and f.endswith('usd'):
                f = 'fee{A}usd'
            elif f.startswith('gaslmtblk'):
                f = 'gaslmtblk'
            elif f.startswith('gaslmttx'):
                f = 'gaslmttx'
            elif f.startswith('gasusedtx'):
                f = 'gasusedtx'
            elif f.startswith('isccont'):
                f = 'isscont'
            record[f] = 'Y'
        records.append(record)

    result_frame = pd.DataFrame.from_records(records).fillna(value='N')
    #result_frame.set_index(keys='symbol', inplace=True)
    result_frame = result_frame.set_index(keys='symbol').T
    latex = result_frame.to_latex()
    print(result_frame.head())




if __name__ == '__main__':
    typer.run(main)
