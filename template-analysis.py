# ============================================================
# DATA ANALYSIS TEMPLATE
# ============================================================
# Event:
# Period:
# Dataset:
#
# Objetivo:
# Template para análise exploratória de dados (EDA),
# distribuição de frequências e estatística descritiva.
# ============================================================


# ============================================================
# IMPORTS
# ============================================================

import os
import webbrowser

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

csv_path = os.path.join(
    BASE_DIR,
    "DataSource",
    "File.csv"
)

REPORT_PATH = os.path.join(
    BASE_DIR,
    "resultado.html"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "Output"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# READ DATA
# ============================================================

print("Reading dataset...")

data = pd.read_csv(
    csv_path,
    header=0,
    sep=","
)

print("Dataset loaded successfully.")


# ============================================================
# DATA CLEANING
# ============================================================

print("\nCleaning dataset...")

# ------------------------------------------------------------
# Remove completely empty rows
# ------------------------------------------------------------

data.dropna(
    how="all",
    inplace=True
)


# ------------------------------------------------------------
# Remove duplicated records
# ------------------------------------------------------------

data.drop_duplicates(
    inplace=True
)


# ------------------------------------------------------------
# Convert possible numeric columns
# ------------------------------------------------------------

numeric_columns = [
    "Column_1",
    "Column_2",
    "Column_3"
]

for column in numeric_columns:

    if column in data.columns:

        data[column] = pd.to_numeric(
            data[column],
            errors="coerce"
        )


# ============================================================
# DATASET INFORMATION
# ============================================================

total_records = len(data)

total_variables = len(data.columns)

numeric_data = data.select_dtypes(
    include=np.number
)

categorical_data = data.select_dtypes(
    exclude=np.number
)


# ============================================================
# DATA QUALITY
# ============================================================

missing_values = data.isnull().sum()

quality_table = pd.DataFrame({

    "Missing Values": missing_values,

    "Percentage (%)":
        (
            missing_values
            / total_records
            * 100
        ).round(2)

})


# ============================================================
# DESCRIPTIVE STATISTICS
# ============================================================

print("\nCalculating descriptive statistics...")

if not numeric_data.empty:

    descriptive_statistics = numeric_data.describe().T

    descriptive_statistics["median"] = (
        numeric_data.median()
    )

    descriptive_statistics["variance"] = (
        numeric_data.var()
    )

    descriptive_statistics["skewness"] = (
        numeric_data.skew()
    )

    descriptive_statistics["kurtosis"] = (
        numeric_data.kurtosis()
    )

    # Reorganize columns

    descriptive_statistics = (
        descriptive_statistics[
            [
                "count",
                "mean",
                "median",
                "std",
                "min",
                "25%",
                "50%",
                "75%",
                "max",
                "variance",
                "skewness",
                "kurtosis"
            ]
        ]
    )

    descriptive_statistics = (
        descriptive_statistics.round(4)
    )

else:

    descriptive_statistics = pd.DataFrame()


# ============================================================
# FREQUENCY DISTRIBUTION
# ============================================================
#
# Para cada variável:
#
# Frequência absoluta  = quantidade de ocorrências
# Frequência relativa  = percentual de ocorrências
#
# ============================================================


def frequency_distribution(series):

    """
    Calculates absolute and relative frequency.
    """

    frequency = (
        series
        .value_counts(dropna=False)
        .sort_index()
    )

    relative_frequency = (
        frequency
        / frequency.sum()
        * 100
    )

    cumulative_frequency = (
        frequency
        .cumsum()
    )

    cumulative_percentage = (
        relative_frequency
        .cumsum()
    )

    table = pd.DataFrame({

        "Frequency": frequency,

        "Percentage (%)":
            relative_frequency.round(2),

        "Cumulative Frequency":
            cumulative_frequency,

        "Cumulative (%)":
            cumulative_percentage.round(2)

    })

    return table


# ============================================================
# FREQUENCY TABLES
# ============================================================

frequency_tables = {}

for column in data.columns:

    frequency_tables[column] = (
        frequency_distribution(
            data[column]
        )
    )


# ============================================================
# HISTOGRAMS AND NORMAL DISTRIBUTION
# ============================================================

print("\nGenerating histograms...")

histogram_paths = {}

for column in numeric_data.columns:

    series = numeric_data[column].dropna()

    # Skip empty variables

    if series.empty:
        continue

    mean = series.mean()

    std = series.std()

    median = series.median()

    q1 = series.quantile(0.25)

    q3 = series.quantile(0.75)

    plt.figure(
        figsize=(12, 7)
    )

    # --------------------------------------------------------
    # Histogram
    # --------------------------------------------------------

    count, bins, ignored = plt.hist(

        series,

        bins=20,

        density=True,

        alpha=0.5,

        edgecolor="black",

        label="Histogram"

    )

    # --------------------------------------------------------
    # Gaussian distribution
    # --------------------------------------------------------

    if std > 0:

        x = np.linspace(

            bins.min(),

            bins.max(),

            500

        )

        gaussian = (

            1
            / (
                std
                * np.sqrt(2 * np.pi)
            )
        ) * np.exp(

            -(
                (x - mean) ** 2
            )
            / (
                2 * std ** 2
            )

        )

        plt.plot(

            x,

            gaussian,

            linewidth=3,

            label="Gaussian Distribution"

        )

    # --------------------------------------------------------
    # Mean
    # --------------------------------------------------------

    plt.axvline(

        mean,

        linestyle="-",

        linewidth=2,

        label=f"Mean ({mean:.2f})"

    )

    # --------------------------------------------------------
    # Median
    # --------------------------------------------------------

    plt.axvline(

        median,

        linestyle="--",

        linewidth=2,

        label=f"Median ({median:.2f})"

    )

    # --------------------------------------------------------
    # Q1
    # --------------------------------------------------------

    plt.axvline(

        q1,

        linestyle=":",

        linewidth=2,

        label=f"Q1 ({q1:.2f})"

    )

    # --------------------------------------------------------
    # Q3
    # --------------------------------------------------------

    plt.axvline(

        q3,

        linestyle=":",

        linewidth=2,

        label=f"Q3 ({q3:.2f})"

    )

    # --------------------------------------------------------
    # Standard deviation
    # --------------------------------------------------------

    if std > 0:

        plt.axvline(

            mean - std,

            linestyle="-.",

            linewidth=1.5,

            label="-1σ"

        )

        plt.axvline(

            mean + std,

            linestyle="-.",

            linewidth=1.5,

            label="+1σ"

        )

        plt.axvline(

            mean - 2 * std,

            linestyle="--",

            linewidth=1.5,

            label="-2σ"

        )

        plt.axvline(

            mean + 2 * std,

            linestyle="--",

            linewidth=1.5,

            label="+2σ"

        )

    # --------------------------------------------------------
    # Chart
    # --------------------------------------------------------

    plt.title(
        f"Distribution of {column}"
    )

    plt.xlabel(
        column
    )

    plt.ylabel(
        "Density"
    )

    plt.grid(
        True
    )

    plt.legend()

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    safe_name = (
        column
        .replace(" ", "_")
        .replace("/", "_")
        .replace("\\", "_")
    )

    histogram_path = os.path.join(

        OUTPUT_DIR,

        f"hist_{safe_name}.png"

    )

    plt.savefig(

        histogram_path,

        dpi=200,

        bbox_inches="tight"

    )

    plt.close()

    histogram_paths[column] = (
        histogram_path
    )


# ============================================================
# HTML REPORT
# ============================================================

print("\nGenerating HTML report...")

with open(

    REPORT_PATH,

    "w",

    encoding="utf-8"

) as f:

    # --------------------------------------------------------
    # HTML HEADER
    # --------------------------------------------------------

    f.write("""
<!DOCTYPE html>

<html>

<head>

<meta charset="utf-8">

<title>Data Analysis Report</title>

<style>

body {

    font-family: Arial, sans-serif;

    margin: 40px;

    line-height: 1.5;

}

h1 {

    color: darkblue;

}

h2 {

    color: #333333;

    border-bottom: 1px solid #cccccc;

    padding-bottom: 5px;

}

h3 {

    color: #555555;

}

table {

    border-collapse: collapse;

    width: 100%;

    margin-bottom: 25px;

}

table, th, td {

    border: 1px solid #999999;

}

th {

    background: #dddddd;

}

td, th {

    padding: 6px;

    text-align: center;

}

img {

    max-width: 100%;

    height: auto;

    margin-bottom: 30px;

}

.section {

    margin-bottom: 40px;

}

</style>

</head>

<body>
""")

    # --------------------------------------------------------
    # TITLE
    # --------------------------------------------------------

    f.write(
        "<h1>Data Analysis Report</h1>"
    )

    f.write(
        "<hr>"
    )

    # --------------------------------------------------------
    # DATASET SUMMARY
    # --------------------------------------------------------

    f.write(
        "<div class='section'>"
    )

    f.write(
        "<h2>Dataset Summary</h2>"
    )

    f.write(
        f"<b>Total records:</b> "
        f"{total_records}<br>"
    )

    f.write(
        f"<b>Total variables:</b> "
        f"{total_variables}<br>"
    )

    f.write(
        f"<b>Numeric variables:</b> "
        f"{len(numeric_data.columns)}<br>"
    )

    f.write(
        f"<b>Categorical variables:</b> "
        f"{len(categorical_data.columns)}"
    )

    f.write(
        "</div>"
    )

    # --------------------------------------------------------
    # DATASET INFORMATION
    # --------------------------------------------------------

    f.write(
        "<div class='section'>"
    )

    f.write(
        "<h2>Dataset Information</h2>"
    )

    info = pd.DataFrame({

        "Column":
            data.columns,

        "Data Type":
            data.dtypes.astype(str),

        "Non-Null":
            data.notnull().sum(),

        "Missing":
            data.isnull().sum(),

        "Unique":
            data.nunique()

    })

    f.write(
        info.to_html(
            index=False
        )
    )

    f.write(
        "</div>"
    )

    # --------------------------------------------------------
    # DATA QUALITY
    # --------------------------------------------------------

    f.write(
        "<div class='section'>"
    )

    f.write(
        "<h2>Data Quality</h2>"
    )

    f.write(
        quality_table.to_html()
    )

    f.write(
        "</div>"
    )

    # --------------------------------------------------------
    # DESCRIPTIVE STATISTICS
    # --------------------------------------------------------

    f.write(
        "<div class='section'>"
    )

    f.write(
        "<h2>Descriptive Statistics</h2>"
    )

    if not descriptive_statistics.empty:

        f.write(
            descriptive_statistics.to_html()
        )

    else:

        f.write(
            "<p>"
            "No numeric variables available."
            "</p>"
        )

    f.write(
        "</div>"
    )

    # --------------------------------------------------------
    # FREQUENCY DISTRIBUTIONS
    # --------------------------------------------------------

    f.write(
        "<div class='section'>"
    )

    f.write(
        "<h2>Frequency Distributions</h2>"
    )

    for column, table in frequency_tables.items():

        f.write(
            f"<h3>{column}</h3>"
        )

        f.write(
            table.to_html()
        )

    f.write(
        "</div>"
    )

    # --------------------------------------------------------
    # HISTOGRAMS
    # --------------------------------------------------------

    f.write(
        "<div class='section'>"
    )

    f.write(
        "<h2>Numerical Variable Distributions</h2>"
    )

    for column, path in histogram_paths.items():

        filename = os.path.basename(path)

        f.write(
            f"<h3>{column}</h3>"
        )

        f.write(
            f'<img src="Output/{filename}">'
        )

    f.write(
        "</div>"
    )

    # --------------------------------------------------------
    # COMPLETE DATASET
    # --------------------------------------------------------

    f.write(
        "<div class='section'>"
    )

    f.write(
        "<h2>Complete Dataset</h2>"
    )

    f.write(
        data.to_html(
            index=False
        )
    )

    f.write(
        "</div>"
    )

    # --------------------------------------------------------
    # FOOTER
    # --------------------------------------------------------

    f.write("""
<hr>

<p>
Generated automatically using Python, Pandas and Matplotlib.
</p>

</body>

</html>
""")


# ============================================================
# OPEN HTML REPORT
# ============================================================

webbrowser.open(
    "file://" + REPORT_PATH
)


# ============================================================
# CONSOLE OUTPUT
# ============================================================

print("\n========================================")
print("DATASET INFORMATION")
print("========================================")

print(
    f"Records: {total_records}"
)

print(
    f"Variables: {total_variables}"
)

print(
    f"Numeric variables: "
    f"{len(numeric_data.columns)}"
)

print(
    f"Categorical variables: "
    f"{len(categorical_data.columns)}"
)


print("\n========================================")
print("MISSING VALUES")
print("========================================")

print(
    quality_table
)


print("\n========================================")
print("DESCRIPTIVE STATISTICS")
print("========================================")

if not descriptive_statistics.empty:

    print(
        descriptive_statistics
    )

else:

    print(
        "No numeric variables."
    )


print("\n========================================")
print("FREQUENCY DISTRIBUTIONS")
print("========================================")

for column, table in frequency_tables.items():

    print(
        f"\n--- {column} ---"
    )

    print(
        table
    )


print("\n========================================")
print("ANALYSIS COMPLETED")
print("========================================")

print(
    f"HTML report: {REPORT_PATH}"
)

print(
    f"Charts directory: {OUTPUT_DIR}"
)
