# ============================================================
# Data Analysis
# Event: Nuclear Weapon Tests
# ============================================================

import os
import webbrowser
import base64
from io import BytesIO

import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

CSV_PATH = os.path.join(
    BASE_DIR,
    "Data",
    "nuclear-weapons-tests.csv"
)

HTML_PATH = os.path.join(
    BASE_DIR,
    "nuclear_weapons_tests_report.html"
)


# ============================================================
# READ CSV
# ============================================================

def read_csv_robust(path):
    """
    Reads the CSV trying common encodings and automatically
    detects comma/semicolon separators.
    """

    encodings = [
        "utf-8",
        "utf-8-sig",
        "cp1252",
        "latin1"
    ]

    last_error = None

    for encoding in encodings:

        try:

            data = pd.read_csv(
                path,
                header=0,
                sep=None,
                engine="python",
                encoding=encoding
            )

            # Some versions of the dataset use semicolon as separator.
            # If pandas still read the entire header as one column, retry
            # explicitly with ';'.
            if len(data.columns) == 1 and ";" in str(data.columns[0]):

                data = pd.read_csv(
                    path,
                    header=0,
                    sep=";",
                    encoding=encoding
                )

            print(
                f"Dataset loaded using encoding: {encoding}"
            )

            return data, encoding

        except UnicodeDecodeError as error:

            last_error = error

    raise UnicodeDecodeError(
        "unknown",
        b"",
        0,
        1,
        f"Could not decode CSV. Last error: {last_error}"
    )


# ============================================================
# NORMALIZE COLUMNS
# ============================================================

def normalize_columns(data):

    data.columns = (
        data.columns
        .astype(str)
        .str.replace("\ufeff", "", regex=False)
        .str.strip()
    )

    return data


# ============================================================
# FIND YEAR COLUMN
# ============================================================

def find_year_column(data):

    candidates = [
        "Ano",
        "ano",
        "Year",
        "year",
        "YEAR"
    ]

    for candidate in candidates:

        if candidate in data.columns:

            if candidate != "Ano":

                data.rename(
                    columns={
                        candidate: "Ano"
                    },
                    inplace=True
                )

            return "Ano"

    raise KeyError(
        "Column 'Ano' was not found.\n\n"
        "Available columns:\n"
        + "\n".join(
            f" - {column}"
            for column in data.columns
        )
        + "\n\n"
        "Possible cause: the CSV delimiter is not being recognized. "
        "The program expects a CSV with comma or semicolon separators."
    )


# ============================================================
# FIND COUNTRY COLUMNS
# ============================================================

def find_country_columns(data):

    candidates = [
        "Estados Unidos",
        "URSS/Rússia",
        "Reino Unido",
        "França",
        "China",
        "Índia",
        "Paquistão",
        "Coréia do Norte"
    ]

    return [
        column
        for column in candidates
        if column in data.columns
    ]


# ============================================================
# CREATE HTML TABLE
# ============================================================

def dataframe_to_html(data):

    return data.to_html(
        index=False,
        classes="data-table",
        border=0,
        justify="center",
        na_rep=""
    )


# ============================================================
# GRAPH TO BASE64
# ============================================================

def figure_to_base64():

    buffer = BytesIO()

    plt.savefig(
        buffer,
        format="png",
        dpi=150,
        bbox_inches="tight"
    )

    plt.close()

    buffer.seek(0)

    return base64.b64encode(
        buffer.read()
    ).decode("utf-8")


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("NUCLEAR WEAPON TESTS - DATA ANALYSIS")
    print("=" * 70)

    # --------------------------------------------------------
    # Verify CSV
    # --------------------------------------------------------

    if not os.path.isfile(CSV_PATH):

        raise FileNotFoundError(
            "\nCSV file not found:\n"
            f"{CSV_PATH}\n\n"
            "Expected structure:\n"
            "Data/\n"
            "  nuclear-weapons-tests.csv\n"
            "  analysis-nuclear-weapon-tests-01.py"
        )

    print(f"\nCSV file: {CSV_PATH}")

    # --------------------------------------------------------
    # Read data
    # --------------------------------------------------------

    data, used_encoding = read_csv_robust(
        CSV_PATH
    )

    # --------------------------------------------------------
    # Normalize column names
    # --------------------------------------------------------

    data = normalize_columns(data)

    print("\nColumns detected:")

    for column in data.columns:

        print(f" - {repr(column)}")

    # --------------------------------------------------------
    # Cleaning
    # --------------------------------------------------------

    rows_original = len(data)

    data.dropna(
        how="all",
        inplace=True
    )

    rows_after_empty = len(data)

    data.drop_duplicates(
        inplace=True
    )

    rows_after_duplicates = len(data)

    # --------------------------------------------------------
    # Identify year
    # --------------------------------------------------------

    find_year_column(data)

    # --------------------------------------------------------
    # Country columns
    # --------------------------------------------------------

    country_columns = find_country_columns(data)

    if not country_columns:

        raise KeyError(
            "No country columns were found.\n\n"
            "Expected columns:\n"
            " - Estados Unidos\n"
            " - URSS/Rússia\n"
            " - Reino Unido\n"
            " - França\n"
            " - China\n"
            " - Índia\n"
            " - Paquistão\n"
            " - Coréia do Norte"
        )

    print("\nCountry columns detected:")

    for country in country_columns:

        print(f" - {country}")

    # --------------------------------------------------------
    # Convert year
    # --------------------------------------------------------

    data["Ano"] = pd.to_numeric(
        data["Ano"],
        errors="coerce"
    )

    # --------------------------------------------------------
    # Convert countries
    # --------------------------------------------------------

    for country in country_columns:

        data[country] = pd.to_numeric(
            data[country],
            errors="coerce"
        )

    # --------------------------------------------------------
    # Remove rows without year
    # --------------------------------------------------------

    data = data[
        data["Ano"].notna()
    ].copy()

    # --------------------------------------------------------
    # Filter event when column exists
    # --------------------------------------------------------

    if "Disaster Type" in data.columns:

        event_values = (
            data["Disaster Type"]
            .astype(str)
            .str.strip()
        )

        nuclear_weapons_data = data[
            event_values == "Nuclear Weapon Tests"
        ].copy()

        # If filtering produces no rows, stop instead
        # of silently analyzing the wrong event.

        if nuclear_weapons_data.empty:

            raise ValueError(
                "The column 'Disaster Type' exists, but no rows "
                "with value 'Nuclear Weapon Tests' were found."
            )

    else:

        print(
            "\nWarning: 'Disaster Type' was not found."
        )

        print(
            "The complete dataset will be analyzed."
        )

        nuclear_weapons_data = data.copy()

    # --------------------------------------------------------
    # Sort by year
    # --------------------------------------------------------

    nuclear_weapons_data.sort_values(
        by="Ano",
        inplace=True
    )

    # --------------------------------------------------------
    # Calculate total from countries
    # --------------------------------------------------------

    nuclear_weapons_data["Total_Calculado"] = (
        nuclear_weapons_data[country_columns]
        .fillna(0)
        .sum(axis=1)
    )

    # Preserve original Total when available,
    # but create it when it does not exist.

    if "Total" not in nuclear_weapons_data.columns:

        nuclear_weapons_data["Total"] = (
            nuclear_weapons_data["Total_Calculado"]
        )

    else:

        nuclear_weapons_data["Total"] = pd.to_numeric(
            nuclear_weapons_data["Total"],
            errors="coerce"
        )

        # Fill missing original totals using calculated totals.

        nuclear_weapons_data["Total"] = (
            nuclear_weapons_data["Total"]
            .fillna(
                nuclear_weapons_data["Total_Calculado"]
            )
        )

    # --------------------------------------------------------
    # Data quality
    # --------------------------------------------------------

    missing_values = (
        nuclear_weapons_data.isnull().sum()
    )

    quality_table = pd.DataFrame(
        {
            "Missing Values": missing_values,
            "Percentage (%)": (
                missing_values
                / len(nuclear_weapons_data)
                * 100
            ).round(2)
        }
    )

    # --------------------------------------------------------
    # Frequency by year
    # --------------------------------------------------------

    frequency_by_year = (
        nuclear_weapons_data
        .groupby(
            "Ano",
            as_index=False
        )["Total"]
        .sum()
    )

    frequency_by_year.rename(
        columns={
            "Total": "Frequência de Testes"
        },
        inplace=True
    )

    frequency_by_year["Ano"] = (
        frequency_by_year["Ano"]
        .astype(int)
    )

    frequency_by_year[
        "Frequência de Testes"
    ] = (
        frequency_by_year[
            "Frequência de Testes"
        ]
        .round()
        .astype(int)
    )

    # --------------------------------------------------------
    # Frequency by country
    # --------------------------------------------------------

    frequency_by_country = (
        nuclear_weapons_data[
            country_columns
        ]
        .sum()
        .sort_values(
            ascending=False
        )
        .reset_index()
    )

    frequency_by_country.columns = [
        "País",
        "Frequência de Testes"
    ]

    frequency_by_country[
        "Frequência de Testes"
    ] = (
        frequency_by_country[
            "Frequência de Testes"
        ]
        .round()
        .astype(int)
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    total_tests = int(
        frequency_by_country[
            "Frequência de Testes"
        ].sum()
    )

    total_years = int(
        frequency_by_year["Ano"].nunique()
    )

    if not frequency_by_year.empty:

        max_year_row = (
            frequency_by_year
            .loc[
                frequency_by_year[
                    "Frequência de Testes"
                ].idxmax()
            ]
        )

        max_year = int(
            max_year_row["Ano"]
        )

        max_year_frequency = int(
            max_year_row[
                "Frequência de Testes"
            ]
        )

    else:

        max_year = "-"
        max_year_frequency = 0

    if not frequency_by_country.empty:

        max_country_row = (
            frequency_by_country.iloc[0]
        )

        max_country = str(
            max_country_row["País"]
        )

        max_country_frequency = int(
            max_country_row[
                "Frequência de Testes"
            ]
        )

    else:

        max_country = "-"
        max_country_frequency = 0

    # --------------------------------------------------------
    # Graph: frequency by year
    # --------------------------------------------------------

    plt.figure(
        figsize=(14, 7)
    )

    plt.plot(
        frequency_by_year["Ano"],
        frequency_by_year[
            "Frequência de Testes"
        ],
        marker="o",
        linewidth=2
    )

    plt.title(
        "Distribuição de Frequência de Testes "
        "Nucleares por Ano",
        fontsize=16
    )

    plt.xlabel(
        "Ano",
        fontsize=12
    )

    plt.ylabel(
        "Número de Testes",
        fontsize=12
    )

    plt.grid(
        True,
        alpha=0.3
    )

    plt.tight_layout()

    year_graph_base64 = (
        figure_to_base64()
    )

    # --------------------------------------------------------
    # Graph: frequency by country - PIE CHART
    # --------------------------------------------------------

    plt.figure(
        figsize=(10, 8)
    )

    plt.pie(
        frequency_by_country["Frequência de Testes"],
        labels=frequency_by_country["País"],
        autopct="%1.1f%%",
        startangle=90
    )

    plt.title(
        "Distribuição de Frequência de Testes "
        "Nucleares por País",
        fontsize=16
    )

    plt.axis("equal")

    plt.tight_layout()

    country_graph_base64 = (
        figure_to_base64()
    )

    # --------------------------------------------------------
    # HTML tables
    # --------------------------------------------------------

    all_data_html = dataframe_to_html(
        nuclear_weapons_data
    )

    quality_html = dataframe_to_html(
        quality_table.reset_index()
        .rename(
            columns={
                "index": "Coluna"
            }
        )
    )

    frequency_year_html = dataframe_to_html(
        frequency_by_year
    )

    frequency_country_html = dataframe_to_html(
        frequency_by_country
    )

    # --------------------------------------------------------
    # HTML report
    # --------------------------------------------------------

    html_content = f"""
<!DOCTYPE html>
<html lang="pt-BR">

<head>

<meta charset="UTF-8">

<meta name="viewport"
      content="width=device-width, initial-scale=1.0">

<title>
Nuclear Weapon Tests - Data Analysis
</title>

<style>

body {{
    font-family:
        Arial,
        Helvetica,
        sans-serif;

    margin: 0;
    padding: 0;

    background: #f4f6f8;
    color: #222;
}}

header {{
    background: #1f2937;
    color: white;

    padding: 30px;

    margin-bottom: 30px;
}}

.container {{
    width: 95%;
    max-width: 1600px;
    margin: auto;
}}

section {{
    background: white;

    padding: 25px;

    margin-bottom: 30px;

    border-radius: 8px;

    box-shadow:
        0 2px 8px rgba(0,0,0,0.08);
}}

h1 {{
    margin: 0;
}}

h2 {{
    color: #1f2937;

    border-bottom:
        2px solid #ddd;

    padding-bottom: 10px;
}}

.summary {{
    display: grid;

    grid-template-columns:
        repeat(
            auto-fit,
            minmax(220px, 1fr)
        );

    gap: 20px;
}}

.card {{
    background: #f8fafc;

    padding: 20px;

    border-radius: 8px;

    text-align: center;

    border:
        1px solid #e5e7eb;
}}

.card .number {{
    font-size: 30px;
    font-weight: bold;

    margin-top: 10px;
}}

.table-container {{
    overflow-x: auto;

    overflow-y: auto;

    max-height: 700px;

    border:
        1px solid #ddd;
}}

table.data-table {{
    width: 100%;

    border-collapse:
        collapse;

    font-size: 14px;
}}

table.data-table th {{
    background: #374151;

    color: white;

    padding: 10px;

    text-align: center;

    position: sticky;

    top: 0;
}}

table.data-table td {{
    padding: 8px;

    border-bottom:
        1px solid #ddd;

    text-align: center;
}}

table.data-table tr:nth-child(even) {{
    background: #f9fafb;
}}

table.data-table tr:hover {{
    background: #eef2ff;
}}

.graph {{
    text-align: center;
}}

.graph img {{
    max-width: 100%;
    height: auto;
}}

footer {{
    text-align: center;

    color: #666;

    padding: 30px;
}}

.info {{
    background: #f8fafc;

    border-left:
        4px solid #374151;

    padding: 15px;

    margin: 15px 0;
}}

</style>

</head>

<body>

<header>

<div class="container">

<h1>
Nuclear Weapon Tests
</h1>

<p>
Data Analysis Report
</p>

<p>
Distribuição de frequência por ano e por país
</p>

</div>

</header>

<div class="container">

<section>

<h2>
1. Resumo da Análise
</h2>

<div class="summary">

<div class="card">

<div>Total de testes</div>

<div class="number">
{total_tests}
</div>

</div>

<div class="card">

<div>Anos analisados</div>

<div class="number">
{total_years}
</div>

</div>

<div class="card">

<div>Ano com maior frequência</div>

<div class="number">
{max_year}
</div>

<div>
{max_year_frequency} testes
</div>

</div>

<div class="card">

<div>País com maior frequência</div>

<div class="number">
{max_country}
</div>

<div>
{max_country_frequency} testes
</div>

</div>

</div>

<div class="info">

<strong>Codificação utilizada:</strong>
{used_encoding}

<br>

<strong>Registros originais:</strong>
{rows_original}

<br>

<strong>Após remoção de linhas vazias:</strong>
{rows_after_empty}

<br>

<strong>Após remoção de duplicados:</strong>
{rows_after_duplicates}

<br>

<strong>Registros analisados:</strong>
{len(nuclear_weapons_data)}

</div>

</section>


<section>

<h2>
2. Qualidade dos Dados
</h2>

<div class="table-container">

{quality_html}

</div>

</section>


<section>

<h2>
3. Distribuição de Frequência por Ano
</h2>

<p>
Número total de testes nucleares registrados
em cada ano.
</p>

<div class="graph">

<img
src="data:image/png;base64,{year_graph_base64}"
alt="Distribuição de frequência por ano"
>

</div>

<h3>
Tabela de Frequência por Ano
</h3>

<div class="table-container">

{frequency_year_html}

</div>

</section>


<section>

<h2>
4. Distribuição de Frequência por País
</h2>

<p>
Número total de testes nucleares registrados
para cada país.
</p>

<div class="graph">

<img
src="data:image/png;base64,{country_graph_base64}"
alt="Distribuição de frequência por país"
>

</div>

<h3>
Tabela de Frequência por País
</h3>

<div class="table-container">

{frequency_country_html}

</div>

</section>


<section>

<h2>
5. Dados Completos
</h2>

<p>
Todos os registros utilizados na análise após
a limpeza e remoção de duplicidades.
</p>

<div class="table-container">

{all_data_html}

</div>

</section>

</div>

<footer>

Nuclear Weapon Tests — Data Analysis

</footer>

</body>

</html>
"""

    # --------------------------------------------------------
    # Save HTML
    # --------------------------------------------------------

    with open(
        HTML_PATH,
        "w",
        encoding="utf-8"
    ) as html_file:

        html_file.write(
            html_content
        )

    # --------------------------------------------------------
    # Console output
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("ANALYSIS RESULTS")
    print("=" * 70)

    print(
        f"\nTotal number of nuclear tests: "
        f"{total_tests}"
    )

    print(
        f"Number of years analyzed: "
        f"{total_years}"
    )

    print(
        f"Year with highest frequency: "
        f"{max_year} "
        f"({max_year_frequency} tests)"
    )

    print(
        f"Country with highest frequency: "
        f"{max_country} "
        f"({max_country_frequency} tests)"
    )

    print("\nFrequency by year:")
    print(
        frequency_by_year.to_string(
            index=False
        )
    )

    print("\nFrequency by country:")
    print(
        frequency_by_country.to_string(
            index=False
        )
    )

    print("\nHTML report:")
    print(HTML_PATH)

    print("\n" + "=" * 70)
    print("OPENING REPORT IN BROWSER")
    print("=" * 70)

    webbrowser.open(
        "file://"
        + os.path.abspath(HTML_PATH)
    )


# ============================================================
# PROGRAM ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()
