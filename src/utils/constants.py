import os

# Constants

CANDLES = 'candles'
ORDERBOOKS = 'orderbooks'

# Exchanges

BINANCE = 'BINANCE'
HUOBI = 'HUOBI'
OKX = 'OKX'

EXCHANGES = [BINANCE, HUOBI, OKX]

CORRELATION_THRESHOLD = 0.9

PCA_VARIANCE_THRESHOLD = 0.95

# Paths

ROOT_DIR = os.path.dirname(os.getcwd())

PROCESSED_DATA_PATH = os.path.join(ROOT_DIR, "data", "processed")
EXTERNAL_DATA_PATH = os.path.join(ROOT_DIR, "data", "external")
INTERIM_DATA_PATH = os.path.join(ROOT_DIR, "data", "interim")

GROUND_TRUTH_PATH = os.path.join(EXTERNAL_DATA_PATH, 'ground_truth', 'BTC.parquet')
GROUND_TRUTH_SUMMARY = os.path.join(EXTERNAL_DATA_PATH, 'ground_truth', 'ground_truth_summary.parquet')

FIGURE_PATH = os.path.join(ROOT_DIR, 'reports', 'figures')

CV_FOLDS = 5
