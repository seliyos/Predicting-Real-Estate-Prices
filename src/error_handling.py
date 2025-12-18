import logging
import sys
from functools import wraps
import traceback

# setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('pipeline.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


# custom exceptions
class DataLoadError(Exception):
    pass

class DataProcessingError(Exception):
    pass

class ModelError(Exception):
    pass


# decorator for catching errors
def handle_errors(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.error(f"error in {func.__name__}: {str(e)}")
            logger.error(traceback.format_exc())
            raise
    return wrapper


# safe data loading
@handle_errors
def load_data_safe(filepath):
    import pandas as pd
    import os
    
    if not os.path.exists(filepath):
        raise DataLoadError(f"file not found: {filepath}")
    
    if not filepath.endswith('.csv'):
        raise DataLoadError(f"only csv supported: {filepath}")
    
    try:
        df = pd.read_csv(filepath)
        logger.info(f"loaded {filepath}: {df.shape[0]} rows, {df.shape[1]} cols")
        
        if df.empty:
            raise DataLoadError("data is empty")
        
        return df
        
    except pd.errors.EmptyDataError:
        raise DataLoadError("csv is empty")
    except pd.errors.ParserError as e:
        raise DataLoadError(f"error parsing csv: {str(e)}")


# validate missing values
@handle_errors  
def check_missing_values(df):
    missing = df.isnull().sum()
    total_missing = missing.sum()
    
    if total_missing > 0:
        logger.warning(f"found {total_missing} missing values")
        high_missing = missing[missing > len(df) * 0.5]
        if len(high_missing) > 0:
            logger.warning(f"columns with >50% missing: {high_missing.index.tolist()}")
    
    return df


# validate shapes match
def validate_shapes(X_train, y_train, X_test=None, y_test=None):
    if len(X_train) != len(y_train):
        raise DataProcessingError(f"train shape mismatch: {len(X_train)} vs {len(y_train)}")
    
    if X_test is not None and y_test is not None:
        if len(X_test) != len(y_test):
            raise DataProcessingError(f"test shape mismatch: {len(X_test)} vs {len(y_test)}")
        
        if X_train.shape[1] != X_test.shape[1]:
            raise DataProcessingError(f"feature mismatch: train has {X_train.shape[1]}, test has {X_test.shape[1]}")
    
    logger.info("shape validation passed")


# check for invalid predictions
def validate_predictions(y_pred, allow_negative=False):
    import numpy as np
    
    if y_pred is None or len(y_pred) == 0:
        raise ModelError("predictions are empty")
    
    if np.any(np.isnan(y_pred)):
        raise ModelError("predictions contain NaN")
    
    if np.any(np.isinf(y_pred)):
        raise ModelError("predictions contain inf")
    
    if not allow_negative and np.any(y_pred < 0):
        logger.warning(f"found {np.sum(y_pred < 0)} negative predictions")
        return np.maximum(y_pred, 0)
    
    return y_pred


# safe model training
@handle_errors
def train_model_safe(model, X_train, y_train, name="model"):
    if not hasattr(model, 'fit'):
        raise ModelError("model doesn't have fit method")
    
    logger.info(f"training {name}...")
    model.fit(X_train, y_train)
    logger.info(f"{name} training done")
    
    return model


# safe model evaluation  
@handle_errors
def evaluate_model_safe(model, X_test, y_test):
    if not hasattr(model, 'predict'):
        raise ModelError("model doesn't have predict method")
    
    y_pred = model.predict(X_test)
    y_pred = validate_predictions(y_pred)
    
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    import numpy as np
    
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    return {'rmse': rmse, 'mae': mae, 'r2': r2}


# wrapper for the full pipeline
@handle_errors
def run_pipeline_safe(train_path, test_path=None):
    from src.data_preprocessing import preprocess_pipeline
    
    logger.info("starting pipeline")
    
    try:
        X_train, y_train, X_test, y_test, encoders, scaler = preprocess_pipeline(
            train_path, test_path, scale=True
        )
        
        validate_shapes(X_train, y_train, X_test, y_test)
        check_missing_values(X_train)
        
        logger.info("pipeline completed successfully")
        
        return X_train, y_train, X_test, y_test, encoders, scaler
        
    except Exception as e:
        logger.error(f"pipeline failed: {str(e)}")
        raise
