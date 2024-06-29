Big Data case study for CS - AI MsC @ UniBA

# Bitcoin Forecasting Project

Welcome to the Bitcoin Forecasting Project.

This project analyzes BTC market data from BINANCE, HUOBI, and OKX exchanges to predict whether the next "close" value will be higher, similar, or lower than the previous one. We employ various machine learning techniques to process and interpret minute-by-minute candles and orderbook data.

## What's Inside?

- **Data Prep:** Cleaned and integrated candles and orderbook data with detailed feature engineering.
- **Modeling:** Utilized RandomForestClassifier with a comprehensive pipeline, including scaling, PCA, and SMOTEENN for handling imbalanced classes.
- **Evaluation:** Detailed evaluation using learning curves, confusion matrices, and feature importance, focusing on balancing bias and variance.

## Highlights

- **Unified Datasets:** Integrating candles and orderbook data provided enhanced predictions for HUOBI and OKX.
- **Comparative Analysis:** Employed bootstrap methods and t-tests to compare model performance across exchanges.
- **Future Work:** Potential to explore advanced models, real-time data processing, and dynamic trading strategies.

## Getting Started

This project uses a cookiecutter for data science template. To install the project, use the following command:

```bash
pip install -e .
```

### Data Requirements

Please note that this project requires data provided by the BINANCE, HUOBI, and OKX exchanges. However, the project is easily customizable to work with data from other sources.

### Notebooks

- **Data Preparation:** Execute the `data_preparation` notebook first to clean and integrate the data.
- **Main Notebook:** After data preparation, execute the `main` notebook to train and evaluate the models.

Feel free to explore the code and contribute.


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
