import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler


def load_data(train_path, test_path=None):
    """Load training and optional test data"""
    train_df = pd.read_csv(train_path)

    if test_path:
        test_df = pd.read_csv(test_path)
        return train_df, test_df

    return train_df


def handle_missing_values(df):
    """
    Handle missing values in the dataset
    - Drop columns with >50% missing
    - Fill numerical with median
    - Fill categorical with 'None'
    """
    df = df.copy()

    # Drop columns with >50% missing values
    missing_pct = df.isnull().sum() / len(df)
    cols_to_drop = missing_pct[missing_pct > 0.5].index
    df = df.drop(columns=cols_to_drop)

    # Separate numerical and categorical columns
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    categorical_cols = df.select_dtypes(include=['object']).columns

    # Fill missing values
    for col in numerical_cols:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].median())

    for col in categorical_cols:
        if df[col].isnull().any():
            df[col] = df[col].fillna('None')

    return df


def encode_categorical_features(df, label_encoders=None):
    """Encode categorical variables using Label Encoding"""
    df = df.copy()
    categorical_cols = df.select_dtypes(include=['object']).columns

    if label_encoders is None:
        label_encoders = {}

    for col in categorical_cols:
        if col in label_encoders:
            df[col] = label_encoders[col].transform(df[col])
        else:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            label_encoders[col] = le

    return df, label_encoders


def feature_engineering(df):
    """Create new engineered features"""
    df = df.copy()

    # Total square footage
    if 'TotalBsmtSF' in df.columns and 'GrLivArea' in df.columns:
        df['TotalSF'] = df['TotalBsmtSF'] + df['GrLivArea']

    # House age
    if 'YearBuilt' in df.columns:
        df['HouseAge'] = 2024 - df['YearBuilt']

    # Years since remodel
    if 'YearRemodAdd' in df.columns:
        df['YearsSinceRemodel'] = 2024 - df['YearRemodAdd']

    # Total bathrooms
    if all(col in df.columns for col in ['FullBath', 'HalfBath', 'BsmtFullBath', 'BsmtHalfBath']):
        df['TotalBathrooms'] = df['FullBath'] + 0.5 * df['HalfBath'] + \
                               df['BsmtFullBath'] + 0.5 * df['BsmtHalfBath']

    # Total porch area
    porch_cols = ['OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch']
    if all(col in df.columns for col in porch_cols):
        df['TotalPorchSF'] = df[porch_cols].sum(axis=1)

    return df


def prepare_features(df, target_col='SalePrice', scale=True, scaler=None):
    """Separate features and target, optionally scale features"""
    df = df.copy()

    # Separate target if it exists
    if target_col in df.columns:
        y = df[target_col]
        X = df.drop(columns=[target_col])
    else:
        y = None
        X = df

    # Drop Id column if exists
    if 'Id' in X.columns:
        X = X.drop(columns=['Id'])

    # Scale features if requested
    if scale:
        if scaler is None:
            scaler = StandardScaler()
            X_scaled = pd.DataFrame(
                scaler.fit_transform(X),
                columns=X.columns,
                index=X.index
            )
        else:
            X_scaled = pd.DataFrame(
                scaler.transform(X),
                columns=X.columns,
                index=X.index
            )
        return X_scaled, y, scaler

    return X, y, None


def preprocess_pipeline(train_path, test_path=None, scale=True):
    """
    Complete preprocessing pipeline

    Returns:
        X_train, y_train, X_test, y_test, label_encoders, scaler
        (X_test and y_test are None if test_path not provided)
    """
    # Load data
    if test_path:
        train_df, test_df = load_data(train_path, test_path)
    else:
        train_df = load_data(train_path)
        test_df = None

    # Handle missing values
    train_df = handle_missing_values(train_df)
    if test_df is not None:
        test_df = handle_missing_values(test_df)

    # Feature engineering
    train_df = feature_engineering(train_df)
    if test_df is not None:
        test_df = feature_engineering(test_df)

    # Encode categorical features
    train_df, label_encoders = encode_categorical_features(train_df)
    if test_df is not None:
        test_df, _ = encode_categorical_features(test_df, label_encoders)

    # Prepare features and target
    X_train, y_train, scaler = prepare_features(train_df, scale=scale)

    if test_df is not None:
        X_test, y_test, _ = prepare_features(test_df, scale=scale, scaler=scaler)
        return X_train, y_train, X_test, y_test, label_encoders, scaler

    return X_train, y_train, None, None, label_encoders, scaler