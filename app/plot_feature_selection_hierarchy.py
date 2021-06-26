import typer
from cryptoml.util.shap import shap
import pandas as pd
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.express as px
import os
import yaml


def load_hierarchy(filename: str, window=10):
    with open(filename, 'r') as f:
        data = yaml.load(f)
    base = data['features']

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
                            'name': f_name
                        })
                else:
                    result.append({
                        'category': category,
                        'subgroup': subgroup,
                        'name': feature
                    })
    return result


def main(dataset: str, target: str):
    hierarchy = load_hierarchy(f"{dataset}_{target}_feature_hierarchy.yml")
    hdf = pd.DataFrame(hierarchy)
    equal_value = 1 / hdf.shape[0]
    new_df = hdf[['category', 'subgroup']].drop_duplicates()
    new_df['importance'] = equal_value
    fig_sunburst = px.sunburst(
        new_df,
        path=['category', 'subgroup'],
        values='importance',
    )
    fig_sunburst.update_traces(insidetextorientation='radial')
    fig_sunburst.update_layout(
        autosize=False,
        margin=dict(
            l=30,
            r=30,
            b=30,
            t=30,
            pad=4
        )
    )
    fig_sunburst.show()
    fig_sunburst.write_image("images/feature_selection/feature_hierarchy.png", width=600, height=600, scale=1)

if __name__ == '__main__':
    typer.run(main)
