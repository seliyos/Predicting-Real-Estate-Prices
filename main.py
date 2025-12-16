from src.data_preprocessing import preprocess_pipeline

def main():
    print("HOUSE PRICE PREDICTION - AMES HOUSING DATASET")

    # Load and preprocess data
    print("\n1. Loading and preprocessing data...")
    X_train, y_train, _, _, encoders, scaler = preprocess_pipeline(
        'data/train.csv',
        scale=True
    )
    print(f"Data ready: {X_train.shape[0]} samples, {X_train.shape[1]} features")

    # TODO: Train models (uncomment when models.py is implemented)

    # TODO: Evaluate models (uncomment when evaluation.py is implemented)

    # TODO: Compare models


if __name__ == "__main__":
    main()