import os



# Constants

CANDLES = 'candles'
ORDERBOOK = 'orderbook'
UNIFIED = 'unified'

# Exchanges

BINANCE = 'BINANCE'
HUOBI = 'HUOBI'
OKX = 'OKX'
ALL = 'ALL'

LAGS = [1, 2, 3, 4]

DATA_TYPES = [CANDLES, ORDERBOOK, UNIFIED]
EXCHANGES = [BINANCE, HUOBI, OKX]

TEST_SIZE = 0.2
RANDOM_STATE = 42

CORRELATION_THRESHOLD = 0.9
PCA_VARIANCE_THRESHOLD = 0.95

# Paths

ROOT_DIR = os.path.dirname(os.getcwd())

EXTERNAL_DATA_PATH = os.path.join(ROOT_DIR, "data", "external")
INTERIM_DATA_PATH = os.path.join(ROOT_DIR, "data", "interim")
PROCESSED_DATA_PATH = os.path.join(ROOT_DIR, "data", "processed")

MODELS_DATA_PATH = os.path.join(ROOT_DIR, "models")

GROUND_TRUTH_PATH = os.path.join(EXTERNAL_DATA_PATH, 'ground_truth', 'BTC.parquet')
GROUND_TRUTH_SUMMARY = os.path.join(EXTERNAL_DATA_PATH, 'ground_truth', 'ground_truth_summary.parquet')
GROUND_TRUTH_PROCESSED_PATH = os.path.join(PROCESSED_DATA_PATH, 'ground_truth.pkl')

FIGURE_PATH = os.path.join(ROOT_DIR, 'reports', 'figures')