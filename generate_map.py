
import pandas as pd
import numpy as np
from plotly import express as px

from ruamel import yaml

with open("map_data.yml") as read_file:
    data = yaml.safe_load(read_file)

merged = list()
for key, value in data["locations"].items():
    merged.append((key, value["iso"]))

df = pd.DataFrame(merged, columns=["Institution", "iso_alpha"])
df["filler"] = np.ones(len(merged))

fig = px.choropleth(df, locations="iso_alpha", hover_data=None)

fig.update_layout(
    showlegend=False,
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)'
)

with open("map.html", "w+") as write_file:
    write_file.write(fig.to_html(include_plotlyjs="cdn"))