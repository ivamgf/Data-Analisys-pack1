# ============================================================
# Correlation Analysis - EM-DAT Earthquakes
# ============================================================

import os
import webbrowser

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# READ DATA
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

csv_path = os.path.join(
    BASE_DIR,
    "Data",
    "public-EM-DAT.csv"
)

data = pd.read_csv(csv_path)

# ============================================================
# FILTER EARTHQUAKES
# ============================================================

earthquakes = data[
    data["Disaster Type"] == "Earthquake"
].copy()

# ============================================================
# SELECT VARIABLES
# ============================================================

columns = [
    "Magnitude",
    "Total Deaths",
    "Total Affected",
    "Total Damage ('000 US$)"
]

corr_data = earthquakes[columns].copy()

# ============================================================
# CONVERT TO NUMERIC
# ============================================================

for column in columns:
    corr_data[column] = pd.to_numeric(
        corr_data[column],
        errors="coerce"
    )

# Remove registros incompletos
corr_data.dropna(inplace=True)

# ============================================================
# CORRELATION MATRIX
# ============================================================

correlation = corr_data.corr(method="pearson")

print("\nCorrelation Matrix\n")
print(correlation)

# ============================================================
# HEATMAP
# ============================================================

fig, ax = plt.subplots(figsize=(8, 7), layout="constrained")

image = ax.imshow(
    correlation,
    cmap="coolwarm",
    vmin=-1,
    vmax=1
)

# Rótulos dos eixos
ax.set_xticks(np.arange(len(columns)))
ax.set_yticks(np.arange(len(columns)))

ax.set_xticklabels(
    columns,
    rotation=45,
    ha="right"
)

ax.set_yticklabels(columns)

# Valores da matriz
for i in range(len(columns)):
    for j in range(len(columns)):

        ax.text(
            j,
            i,
            f"{correlation.iloc[i, j]:.2f}",
            ha="center",
            va="center",
            color="black",
            fontsize=11
        )

plt.title("Pearson Correlation Matrix")

plt.colorbar(image)

heatmap_path = os.path.join(
    BASE_DIR,
    "correlation_heatmap.png"
)

plt.savefig(
    heatmap_path,
    dpi=200
)

plt.close()

# ============================================================
# SCATTER MATRIX
# ============================================================

scatter_path = os.path.join(
    BASE_DIR,
    "scatter_matrix.png"
)

pd.plotting.scatter_matrix(

    corr_data,

    figsize=(12, 12),

    diagonal="hist"

)

plt.suptitle(
    "Scatter Matrix",
    fontsize=16
)

plt.savefig(
    scatter_path,
    dpi=200
)

plt.close()

# ============================================================
# HTML REPORT
# ============================================================

html_path = os.path.join(
    BASE_DIR,
    "correlation_report.html"
)

with open(
    html_path,
    "w",
    encoding="utf-8"
) as f:

    f.write("""
<html>
<head>
<meta charset="utf-8">

<title>Correlation Report</title>

<style>

body{

    font-family:Arial;
    margin:40px;

}

table{

    border-collapse:collapse;
    width:80%;

}

th,td{

    border:1px solid #999;
    padding:8px;
    text-align:center;

}

th{

    background:#dddddd;

}

</style>

</head>

<body>
""")

    f.write("<h1>Correlation Analysis</h1>")

    f.write("<h2>Variables</h2>")

    f.write("<ul>")

    for column in columns:
        f.write(f"<li>{column}</li>")

    f.write("</ul>")

    f.write("<h2>Correlation Matrix</h2>")

    f.write(correlation.round(3).to_html())

    f.write("<h2>Correlation Heatmap</h2>")

    f.write(
        '<img src="correlation_heatmap.png" width="700">'
    )

    f.write("<h2>Scatter Matrix</h2>")

    f.write(
        '<img src="scatter_matrix.png" width="900">'
    )

    f.write("</body></html>")

# ============================================================
# OPEN REPORT
# ============================================================

webbrowser.open(
    "file://" + html_path
)

print("\nCorrelation report generated successfully.")