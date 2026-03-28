# Online Retail Forecasting

This repository contains an end-to-end workflow for:

- preprocessing raw online retail product data
- clustering products into demand-behavior groups
- training cluster-specific forecasting models
- analyzing results with notebooks and plots
- serving product-level forecast lookup through a Streamlit agent

The project is organized around five main stages:

1. `EDA & Data Extraction`: download original dataset from website, and do EDA
2. `Pre-process`: clean product descriptions and prepare canonical product-family mapping
3. `Pre-modeling`: build clustering features, choose clustering setup, and create train/test forecasting panels
4. `Modeling`: train and compare cluster-specific forecasting models for clusters `C0`, `C1`, `C2`, and `C3`
5. `Agent`: forecast and expose a Streamlit interface for product lookup

## Dataset

This project uses the **Online Retail II** transaction dataset from the UCI Machine Learning Repository.

- Source: `https://archive.ics.uci.edu/static/public/502/online+retail+ii.zip`
- Raw format: Excel workbook with two sheets (`Year 2009-2010`, `Year 2010-2011`)
- Core fields used through the pipeline include invoice date, stock code, description, quantity/amount signals, and customer/country metadata
- Modeling unit in this repo: product-family daily demand time series derived from cleaned transactional records

## Architecture

```text
raw Excel
  -> product cleaning / family mapping
  -> parquet + cleaned workbook
  -> clustering features and cluster assignments
  -> train/test daily forecasting panels
  -> cluster-specific forecasting models
  -> prediction parquet files + evaluation plots
  -> agent artifacts
  -> Streamlit app
```

At a high level:

- `product_preprocess.py` and `product_classifier.py` standardize product names
- clustering notebooks create feature tables and cluster assignments
- forecasting scripts train one or more models per cluster
- analysis utilities and notebooks generate evaluation tables and SVG plots
- `agent/` reads the forecast outputs and serves them in a user-facing app

## Environment Setup

Environment files are kept in the repo:

- [`environment.yml`]: cross-platform environment (support both Mac and Windows)

Recommended setup:

```bash
cd OnlineRetail
conda env create -f environment.yml
conda activate forecasting-retail
```

If the environment already exists:

```bash
conda env update -f environment.yml --prune
conda activate forecasting-retail
```

For notebook execution from the terminal, this project works most reliably with:

```bash
PYTHONNOUSERSITE=1 python -s -m nbconvert --to notebook --execute --inplace --ExecutePreprocessor.kernel_name=forecasting-retail your_notebook.ipynb
```

## Repository Layout

### Top-level folders

- [`data/`]: canonical data inputs and intermediate outputs
  - `raw/`: original Excel / zip files
  - `clustering/`: clustering features, assignments, and clustering outputs
  - `forecasting/`: splitted train/test daily panels and cluster-level prediction outputs
- [`images/`]: saved SVG/PNG figures for clustering and forecasting analysis
  - `clustering/`: clustering diagnostics and embeddings
  - `modeling/`: raw cluster time-series plots
  - `c0_result/`, `c1_results/`, `c2_result/`, `c3_results/`: model evaluation outputs by cluster
  - `agent/`: screenshots and agent-related visuals
- [`agent/`]: Streamlit app, forecast lookup tools, prompt logic, and tests

## File Map By Stage

### EDA & Data Extraction

- [`data_extraction_eda.ipynb`]
  - Purpose: download raw data from website, unzip to seperate Excel sheets, deliver exploratory tables and data understanding for subsequent cleaning
  - Input: `https://archive.ics.uci.edu/static/public/502/online+retail+ii.zip`
  - Output: [`data/raw/online_retail_II.xlsx`]

### Pre-process

- [`product_preprocess.py`]
  - Purpose: reads raw Excel sheets, cleans product description fields, and writes a cleaned workbook
  - Input: [`data/raw/online_retail_II.xlsx`]
  - Output: [`data/online_retail_II_cleaned.xlsx`]

- [`product_classifier.py`]
  - Purpose: uses the OpenAI API to map raw descriptions to normalized product families and variant hints
  - Input: [`data/online_retail_II_cleaned.xlsx`], [`desc2fam_checkpoint.jsonl`]
  - Output: [`desc2fam_checkpoint.jsonl`], [`data/Year_2009-2010.parquet`], ['data/Year_2010-2011.parquet']

- [`product_postprocess.py`]
  - Purpose: post-processes classified product outputs into finalized product-family representations
  - Input/Output: [`data/Year_2009-2010.parquet`], ['data/Year_2010-2011.parquet']

### Pre-modeling - Clustering

- [`cluster_analysis.ipynb`]
  - Purpose: train/test split on 70-30 ratio, clustering experiments, K selection, stability analysis, UMAP plots, and cluster profiling
  - Key outputs:
    - [`data/clustering/train_df.parquet`], [`data/clustering/test_df.parquet`]
    - [`data/clustering/k_selection_metrics.parquet`]
    - [`data/clustering/clusters_3models.parquet`]
    - [`data/clustering/cluster_size_share_3models.parquet`]
    - [`data/clustering/cluster_stability_ari_3models.parquet`]
    - [`images/clustering/`]

- [`merge_clustering_results.py`]
  - Purpose: aggregate train/test dataset by day, ready for downstream modeling
  - Input: [`data/clustering/train_df.parquet`], [`data/clustering/test_df.parquet`]
  - Output: [`data/forecasting/train_daily.parquet`], ['data/forecasting/test_daily.parquet']

### Modeling

#### Cluster C0

- Output: ['data/forecasting/c0_prediction.parquet']

- [`c0_m1.py`]
  - Stage: modeling
  - Model: TSB baseline for intermittent demand `C0`
  - Output: predictions and tuning summaries

- [`c0_m2.py`]
  - Stage: modeling
  - Model: two-stage LightGBM production model for `C0`
  - Output: prediction parquet plus artifacts consumed by comparison notebooks and the agent

#### Cluster C1

- Output: ['data/forecasting/c1_prediction.parquet']

- [`c1_forecasting.py`]
  - Stage: modeling
  - Purpose: cluster `C1` forecasting pipeline (naive7 baseline vs leak-safe ZINB)
  - Output: `data/forecasting/c1_prediction.parquet` and analysis-ready metric artifacts

#### Cluster C2

- Output: ['data/forecasting/c2_prediction.parquet']

- [`c2_m1.py`]
  - Stage: modeling
  - Model: Holt-Winters / ETS baseline for cluster `C2`
  - Output: predictions and tuning summaries

- [`c2_m2.py`]
  - Stage: modeling
  - Model: global LightGBM production model for cluster `C2`
  - Output: prediction parquet and trained-artifact tables for analysis and agent

#### Cluster C3

- Output: ['data/forecasting/c3_prediction.parquet']

- [`c3_forecasting.py`]
  - Stage: modeling
  - Purpose: cluster `C3` forecasting pipeline (naive7 baseline vs leak-safe two-stage HGB)
  - Output: `data/forecasting/c3_prediction.parquet` and period-level metric artifacts

#### Modeling notebooks

- [`forecasting_0&2.ipynb`]
  - Purpose: end-to-end run and comparison notebook for clusters `C0` and `C2`
  - Output:
    - metric tables for `m1` vs `m2` on both `C0` and `C2`
    - period-level evaluation tables
    - boxplots and comparison figures in [`images/c0_result/`] and [`images/c2_result/`]

- [`c1c3_forecasting.ipynb`]
  - Purpose: main comparative-study notebook for `C1` and `C3`; runs final pairwise model comparisons and exports production prediction files used by downstream analysis/agent tooling
  - Output: `data/forecasting/c1_prediction.parquet`, `data/forecasting/c3_prediction.parquet`, plus comparison metrics/plots

- [`c1c3_model_comparison_new.ipynb`]
  - Purpose: extended/alternate comparative-study notebook for `C1` and `C3` using the newer experimental pipelines
  - Output: evaluation plots in [`images/c1_results/`] and [`images/c3_results/`]

#### Helper files

- [`c0c2_analysis.py`]
  - Purpose: evaluation helpers for cluster `C0` and `C2`
  - Includes: pointwise error metrics, per-period evaluation, prediction-frame construction
  - Output: metric bundles, period tables, error distributions used by notebooks and scripts

- [`c1c3_analysis.py`]
  - Purpose: post-forecast analysis for cluster `C1` and `C3`
  - Output: evaluation tables and plots such as rolling error trends, confusion-by-period, residual charts, and boxplots

### Agent

- [`agent/app.py`]
  - Purpose: Streamlit frontend for product-level forecast lookup
  - Output: local web app

- [`agent/build_assets.py`]
  - Purpose: converts forecast outputs into validated agent-ready assets
  - Output: files in [`agent/artifacts/`]

- [`agent/config.py`]
  - Purpose: path, environment-variable, and model-routing configuration

- [`agent/query_parser.py`]
  - Purpose: parses product lookup requests; optionally uses OpenAI for query normalization

- [`agent/tools.py`]
  - Purpose: core lookup tools used by the app

- [`agent/registry.py`]
  - Purpose: product registry build / resolution logic

- [`agent/unavailable_products.py`]
  - Purpose: handles unavailable-product messaging and fallback behavior

- [`agent/prompts.py`]
  - Purpose: system prompts and summary payload templates for the app

- [`agent/tests/`]
  - Purpose: unit tests for parser, tools, app helpers, and asset validation

## Results

### Prediction files

These are the main structured outputs from modeling:

- [`data/forecasting/c0_prediction.parquet`]
- [`data/forecasting/c1_prediction.parquet`]
- [`data/forecasting/c2_prediction.parquet`]
- [`data/forecasting/c3_prediction.parquet`]

Each file stores forecast outputs aligned to the project’s train/test forecasting panels.

### Result figures

Cluster-specific visual summaries live here:

- `C0`: [`images/c0_result/`]
- `C1`: [`images/c1_results/`]
- `C2`: [`images/c2_result/`]
- `C3`: [`images/c3_results/`]

Typical plots include:

- actual vs forecast cluster aggregates
- rolling error trend
- occurrence confusion by test period
- positive-demand scatter
- residual / interval-width views
- error decomposition
- period-level epsilon-APE boxplots

### Raw time-series visual context

- [`images/modeling/c0_raw_timeseries.svg`]
- [`images/modeling/c1_raw_timeseries.svg`]
- [`images/modeling/c2_raw_timeseries.svg`]
- [`images/modeling/c3_raw_timeseries.svg`]

## Agent Usage

Create a local `.env` file from `.env.example` and set your OpenAI key:

```env
OPENAI_API_KEY=your_api_key_here
ENABLE_OPENAI_QUERY_PARSER=1
OPENAI_QUERY_PARSER_MODEL=gpt-4o-mini
OPENAI_SUMMARY_MODEL=gpt-4o-mini
```

## Start The Agent

1. Build the validated forecast artifacts (We have already built it, you don't need to run this):

```bash
python -m agent.build_assets --include-future
```

2. Launch the Streamlit app:

```bash
python -m streamlit run agent/app.py
```

3. Open the local URL shown in the terminal, usually `http://localhost:8501`.

## How To Use The Agent

![Agent home](images/agent/agent_home.png)

The agent supports two modes:

- `Future mode`:
  Uses `train + test` history to forecast beyond the test period.
  This is the default and recommended business-facing mode.
- `Evaluation mode`:
  Reproduces the test-period forecast, so the response can include `actual_value`.

Recommended workflow:

1. Open the app.
2. Keep `Future mode` on unless you specifically want backtest results.
3. Enter a product name or stock code in the chat box.
4. Ask for weekly forecasts when possible, because weekly queries are the clearest way to present these models.

Example queries:

```text
Give me a weekly forecast for 12 PENCIL SMALL TUBE WOODLAND for 6 weeks
```
```text
Show me the next 8 weeks for stock code 20973
```

The app returns:
- matched product name and product ID
- cluster assignment
- production model used for that cluster
- forecast table
- forecast chart

## Cluster Routing

The agent does not let the LLM choose the model. Routing is fixed by cluster:

- `C0`: Two-stage LGBM
- `C1`: ZINB
- `C2`: Global LGBM
- `C3`: Two-stage HGB
