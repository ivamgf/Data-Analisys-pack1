# ============================================================
# Data Analysis Natural Disasters
# Event: Earthquake
# Period: 2000-2026
# ============================================================

# Imports
import os
import webbrowser

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import folium
from folium.plugins import HeatMap

# ============================================================
# READ DATA
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

csv_path = os.path.join(
    BASE_DIR,
    "Data",
    "public-EM-DAT-2000-2026.csv"
)

emdat_pack1_data = pd.read_csv(
    csv_path,
    header=0,
    sep=","
)

# ============================================================
# DATA CLEANING
# ============================================================

print("Cleaning dataset...")

# Remove linhas completamente vazias
emdat_pack1_data.dropna(
    how="all",
    inplace=True
)

# Remove registros duplicados
emdat_pack1_data.drop_duplicates(
    inplace=True
)

# Colunas numéricas

numeric_columns = [

    "Magnitude",
    "Latitude",
    "Longitude",

    "Start Year",
    "Start Month",
    "Start Day",

    "Total Deaths",
    "No. Injured",
    "No. Affected",
    "No. Homeless",
    "Total Affected",

    "Total Damage ('000 US$)",
    "Total Damage, Adjusted ('000 US$)"
]

for column in numeric_columns:

    if column in emdat_pack1_data.columns:

        emdat_pack1_data[column] = pd.to_numeric(
            emdat_pack1_data[column],
            errors="coerce"
        )

# ============================================================
# DATA QUALITY
# ============================================================

missing_values = emdat_pack1_data.isnull().sum()

quality_table = pd.DataFrame({

    "Missing Values": missing_values,
    "Percentage (%)":
        (missing_values / len(emdat_pack1_data) * 100).round(2)

})

# ============================================================
# FILTER EARTHQUAKES
# ============================================================

earthquake_data = emdat_pack1_data[
    emdat_pack1_data["Disaster Type"] == "Earthquake"
].copy()

# ============================================================
# HEAT MAP DATA
# ============================================================

heatmap_data = earthquake_data[
    ["Latitude","Longitude","Magnitude"]
].dropna()

heatmap_data = heatmap_data.astype(float)

heatmap_data = heatmap_data.astype(float)

# ============================================================
# FREQUENCY DISTRIBUTIONS
# ============================================================

# Frequência por ano
freq_year = (
    earthquake_data["Start Year"]
    .dropna()
    .astype(int)
    .value_counts()
    .sort_index()
)

# Frequência por país
freq_country = (
    earthquake_data["Country"]
    .fillna("Unknown")
    .value_counts()
    .sort_values(ascending=False)
)

# Frequência por continente
freq_region = (
    earthquake_data["Region"]
    .fillna("Unknown")
    .value_counts()
    .sort_values(ascending=False)
)

# Frequência por sub-região
freq_subregion = (
    earthquake_data["Subregion"]
    .fillna("Unknown")
    .value_counts()
    .sort_values(ascending=False)
)

# ============================================================
# TIME SERIES - NUMBER OF EARTHQUAKES PER YEAR
# ============================================================

time_series = (
    earthquake_data
    .groupby("Start Year")
    .size()
    .sort_index()
)

plt.figure(figsize=(14,6))

plt.plot(
    time_series.index,
    time_series.values,
    color="navy",
    linewidth=2,
    marker="o",
    markersize=4
)

plt.title("Time Series - Number of Earthquakes per Year")

plt.xlabel("Year")

plt.ylabel("Number of Earthquakes")

plt.grid(True, linestyle="--", alpha=0.5)

timeseries_path = os.path.join(
    BASE_DIR,
    "time_series_year.png"
)

plt.tight_layout()

plt.savefig(
    timeseries_path,
    dpi=200,
    bbox_inches="tight"
)

plt.close()

# ============================================================
# LINE CHART - FREQUENCY DISTRIBUTION
# ============================================================

plt.figure(figsize=(12,6))

plt.plot(
    freq_year.index,
    freq_year.values,
    marker="o",
    linewidth=2
)

plt.title("Earthquake Frequency by Year")

plt.xlabel("Year")

plt.ylabel("Number of Earthquakes")

plt.grid(True)

line_path = os.path.join(
    BASE_DIR,
    "frequency_year.png"
)

plt.savefig(
    line_path,
    dpi=200,
    bbox_inches="tight"
)

plt.close()

# ============================================================
# HEAT MAP - LATITUDE / LONGITUDE
# ============================================================

# Centro do mapa
center_lat = heatmap_data["Latitude"].mean()
center_lon = heatmap_data["Longitude"].mean()

heat_map = folium.Map(

    location=[center_lat, center_lon],

    zoom_start=2,

    tiles="CartoDB positron"

)

HeatMap(

    heatmap_data.values,

    radius=10,

    blur=8,

    max_zoom=7

).add_to(heat_map)

heatmap_path = os.path.join(

    BASE_DIR,

    "earthquake_heatmap.html"

)

heat_map.save(
    heatmap_path
)

# ============================================================
# SCATTER MAP
# ============================================================

scatter = folium.Map(

    location=[center_lat, center_lon],

    zoom_start=2,

    tiles="CartoDB positron"

)

for _, row in earthquake_data.dropna(
    subset=["Latitude","Longitude"]
).iterrows():

    folium.CircleMarker(

        location=[

            row["Latitude"],

            row["Longitude"]

        ],

        radius=max(
            2,
            row["Magnitude"]/2
        ),

        popup=f"""
Country: {row['Country']}<br>
Magnitude: {row['Magnitude']}<br>
Year: {int(row['Start Year'])}
""",

        fill=True

    ).add_to(scatter)

scatter.save(

    os.path.join(
        BASE_DIR,
        "earthquake_scatter.html"
    )
)

# ============================================================
# DESCRIPTIVE STATISTICS
# ============================================================

magnitude = earthquake_data["Magnitude"].dropna()

describe_mag = magnitude.describe()

skewness = magnitude.skew()

kurtosis = magnitude.kurtosis()

# ============================================================
# HISTOGRAM + NORMAL DISTRIBUTION
# ============================================================

mean = magnitude.mean()
std = magnitude.std()

median = magnitude.median()

q1 = magnitude.quantile(0.25)

q3 = magnitude.quantile(0.75)

plt.figure(figsize=(12,7))

count, bins, ignored = plt.hist(
    magnitude,
    bins=20,
    density=True,
    alpha=0.5,
    edgecolor="black",
    label="Histogram"
)

x = np.linspace(
    bins.min(),
    bins.max(),
    500
)

gaussian = (
    1/(std*np.sqrt(2*np.pi))
) * np.exp(
    -((x-mean)**2)/(2*std**2)
)

plt.plot(
    x,
    gaussian,
    color="red",
    linewidth=3,
    label="Gaussian Distribution"
)

# Média
plt.axvline(
    mean,
    color="blue",
    linestyle="-",
    linewidth=2,
    label=f"Mean ({mean:.2f})"
)

# Mediana
plt.axvline(
    median,
    color="green",
    linestyle="--",
    linewidth=2,
    label=f"Median ({median:.2f})"
)

# Quartis
plt.axvline(
    q1,
    color="orange",
    linestyle=":",
    linewidth=2,
    label=f"Q1 ({q1:.2f})"
)

plt.axvline(
    q3,
    color="orange",
    linestyle=":",
    linewidth=2,
    label=f"Q3 ({q3:.2f})"
)

# ±1σ
plt.axvline(
    mean-std,
    color="purple",
    linestyle="-.",
    linewidth=2,
    label="-1σ"
)

plt.axvline(
    mean+std,
    color="purple",
    linestyle="-.",
    linewidth=2,
    label="+1σ"
)

# ±2σ
plt.axvline(
    mean-2*std,
    color="brown",
    linestyle="--",
    linewidth=1.5,
    label="-2σ"
)

plt.axvline(
    mean+2*std,
    color="brown",
    linestyle="--",
    linewidth=1.5,
    label="+2σ"
)

plt.title("Earthquake Magnitude Distribution")

plt.xlabel("Magnitude")

plt.ylabel("Density")

plt.grid(True)

plt.legend()

hist_path = os.path.join(
    BASE_DIR,
    "hist_magnitude.png"
)

plt.savefig(
    hist_path,
    dpi=200,
    bbox_inches="tight"
)

plt.close()

# ============================================================
# BOXPLOT - MAGNITUDE BY REGION
# ============================================================

# Remove registros sem região ou magnitude
boxplot_data = earthquake_data[
    ["Region", "Magnitude"]
].dropna()

# Ordena as regiões pela quantidade de registros
order = (
    boxplot_data["Region"]
    .value_counts()
    .index
)

# Cria os dados para o boxplot
data = [
    boxplot_data.loc[
        boxplot_data["Region"] == region,
        "Magnitude"
    ].values
    for region in order
]

plt.figure(figsize=(14,8))

region_stats = (
    earthquake_data
    .groupby("Region")["Magnitude"]
    .describe()
)

plt.boxplot(
    data,
    tick_labels=order,
    orientation="vertical",
    patch_artist=True,
    showmeans=True,
    meanline=True
)

plt.title("Earthquake Magnitude by Region")

plt.xlabel("Region")

plt.ylabel("Magnitude")

plt.xticks(rotation=30)

plt.grid(axis="y", linestyle="--", alpha=0.5)

boxplot_path = os.path.join(
    BASE_DIR,
    "boxplot_region.png"
)

plt.tight_layout()

plt.savefig(
    boxplot_path,
    dpi=200,
    bbox_inches="tight"
)

plt.close()

# ============================================================

# ============================================================
# HTML REPORT
# ============================================================

html_path = os.path.join(

    BASE_DIR,

    "resultado.html"

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

<title>EM-DAT Earthquake Report</title>

<style>

body{

    font-family:Arial;

    margin:40px;

}

table{

    border-collapse:collapse;

    width:100%;

}

table,th,td{

    border:1px solid gray;

}

th{

    background:#dddddd;

}

td,th{

    padding:5px;

    text-align:center;

}

h1{

    color:darkblue;

}

</style>

</head>

<body>

""")

    # --------------------------------------------------------

    f.write("<h1>EM-DAT - Earthquake Analysis</h1>")

    f.write("<hr>")

    f.write(f"<b>Total records:</b> {len(earthquake_data)}<br>")

    f.write(f"<b>Total variables:</b> {len(emdat_pack1_data.columns)}")

    # --------------------------------------------------------

    f.write("<h2>Dataset Information</h2>")

    info = pd.DataFrame({

        "Column": emdat_pack1_data.columns,

        "Data Type": emdat_pack1_data.dtypes.astype(str)

    })

    f.write(info.to_html(index=False))

    # --------------------------------------------------------

    f.write("<h2>Missing Values</h2>")

    f.write(quality_table.to_html())

    # --------------------------------------------------------

    # --------------------------------------------------------

    f.write("<h2>Complete Dataset</h2>")

    f.write(

        earthquake_data.to_html(

            index=False

        )

    )

    # ============================================================
    # YEAR FREQUENCY
    # ============================================================

    f.write("<h2>Frequency Distribution by Year</h2>")

    f.write(
        freq_year
        .to_frame("Frequency")
        .to_html()
    )

    f.write("<br>")
    f.write('<img src="frequency_year.png" width="900">')

    f.write(
        "<p>Temporal evolution of the annual number of earthquakes "
        "with a 5-year moving average.</p>"
    )

    f.write("<h3>Annual Time Series</h3>")

    f.write(
        time_series
        .to_frame("Earthquakes")
        .to_html()
    )

    # ============================================================
    # COUNTRY FREQUENCY
    # ============================================================

    plt.figure(figsize=(14, 8))

    freq_country.head(20).plot(
        kind="bar"
    )

    plt.title("Top 20 Countries - Earthquake Frequency")

    plt.xlabel("Country")

    plt.ylabel("Frequency")

    plt.xticks(rotation=75)

    plt.grid(axis="y")

    country_path = os.path.join(
        BASE_DIR,
        "frequency_country.png"
    )

    plt.tight_layout()

    plt.savefig(
        country_path,
        dpi=200
    )

    plt.close()

    # ============================================================
    # REGION FREQUENCY
    # ============================================================

    plt.figure(figsize=(10, 6))

    freq_region.plot(
        kind="bar"
    )

    plt.title("Earthquake Frequency by Region")

    plt.xlabel("Region")

    plt.ylabel("Frequency")

    plt.grid(axis="y")

    region_path = os.path.join(
        BASE_DIR,
        "frequency_region.png"
    )

    plt.tight_layout()

    plt.savefig(
        region_path,
        dpi=200
    )

    plt.close()

    # ============================================================
    # SUBREGION FREQUENCY
    # ============================================================

    plt.figure(figsize=(14, 8))

    freq_subregion.plot(
        kind="bar"
    )

    plt.title("Earthquake Frequency by Subregion")

    plt.xlabel("Subregion")

    plt.ylabel("Frequency")

    plt.xticks(rotation=75)

    plt.grid(axis="y")

    subregion_path = os.path.join(
        BASE_DIR,
        "frequency_subregion.png"
    )

    plt.tight_layout()

    plt.savefig(
        subregion_path,
        dpi=200
    )

    plt.close()

    # ============================================================
    # COUNTRY
    # ============================================================

    f.write("<h2>Frequency Distribution by Country</h2>")

    f.write(
        freq_country
        .to_frame("Frequency")
        .to_html()
    )

    f.write('<img src="frequency_country.png" width="1000">')

    # ============================================================
    # REGION
    # ============================================================

    f.write("<h2>Frequency Distribution by Region</h2>")

    f.write(
        freq_region
        .to_frame("Frequency")
        .to_html()
    )

    f.write('<img src="frequency_region.png" width="900">')

    # ============================================================
    # SUBREGION
    # ============================================================

    f.write("<h2>Frequency Distribution by Subregion</h2>")

    f.write(
        freq_subregion
        .to_frame("Frequency")
        .to_html()
    )

    f.write('<img src="frequency_subregion.png" width="1100">')

    # --------------------------------------------------------

    f.write("<h2>Magnitude - Descriptive Statistics</h2>")

    f.write(

        describe_mag

        .to_frame("Magnitude")

        .to_html()

    )

    # --------------------------------------------------------

    f.write("<h2>Additional Statistics</h2>")

    f.write("<ul>")

    f.write(f"<li><b>Mean:</b> {mean:.3f}</li>")

    f.write(f"<li><b>Standard Deviation:</b> {std:.3f}</li>")

    f.write(f"<li><b>Skewness:</b> {skewness:.3f}</li>")

    f.write(f"<li><b>Kurtosis:</b> {kurtosis:.3f}</li>")

    f.write("</ul>")

    # --------------------------------------------------------

    f.write("<h2>Histogram with Gaussian Distribution</h2>")

    f.write('<img src="hist_magnitude.png" width="900">')

    f.write("</body></html>")

    # --------------------------------------------------------

    # ============================================================
    # MAGNITUDE PER REGION
    # ============================================================

    f.write("<h2>Magnitude BoxPlot by Region</h2>")

    f.write(
        "<p>Comparação da distribuição das magnitudes "
        "dos terremotos por continente.</p>"
    )

    f.write(
        '<img src="boxplot_region.png" width="1000">'
    )

    f.write("<h2>Magnitude Statistics by Region</h2>")

    f.write(
        region_stats.to_html()
    )

    # --------------------------------------------------------

    # ============================================================
    # HEATMAP
    # ============================================================

    f.write("<h2>Earthquake Heat Map</h2>")

    f.write("""
    <p>
    Interactive heat map showing the spatial concentration
    of earthquakes based on latitude and longitude.
    </p>
    """)

    f.write(
        """
        <iframe
        src="earthquake_heatmap.html"
        width="100%"
        height="700">
        </iframe>
        """
    )

    f.write("<h2>Earthquake Scatter Map</h2>")

    f.write("""
    <iframe
    src="earthquake_scatter.html"
    width="100%"
    height="700">
    </iframe>
    """)

# ============================================================
# OPEN HTML
# ============================================================

webbrowser.open(

    "file://" + html_path

)

print("\nHTML report opened successfully.")

# ============================================================
# CONSOLE OUTPUT
# ============================================================

print("\n============================")
print("Dataset Information")
print("============================")

print(emdat_pack1_data.info())

print("\n============================")
print("Missing Values")
print("============================")

print(quality_table)

print("\n============================")
print("Frequency Distribution")
print("============================")

print(freq_year)

print("\n============================")
print("Describe Magnitude")
print("============================")

print(describe_mag)

print("\n============================")
print("Skewness:", skewness)

print("Kurtosis:", kurtosis)