from src.data_preprocessing import preprocess_pipeline
from src.models import train_all_models
from src.evaluation import evaluate_model, compare_models
from sklearn.model_selection import train_test_split


def main():
    print("HOUSE PRICE PREDICTION - AMES HOUSING DATASET")

    # load data
    print("\n1. Loading and preprocessing data...")
    X_train, y_train, _, _, encoders, scaler = preprocess_pipeline(
        'data/train.csv',
        scale=True
    )
    print(f"Data ready: {X_train.shape[0]} samples, {X_train.shape[1]} features")

    # split for validation
    print("\n2. Splitting data...")
    X_train_split, X_val, y_train_split, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42
    )
    print(f"Train: {X_train_split.shape[0]}, Val: {X_val.shape[0]}")

    # train models
    print("\n3. Training models...")
    models = train_all_models(X_train_split, y_train_split)
    print(f"Done - trained {len(models)} models")

    # evaluate
    print("\n4. Evaluating models...")
    results = {}
    for name, model in models.items():
        metrics = evaluate_model(model, X_val, y_val)
        results[name] = metrics
        print(f"{name}: RMSE={metrics['rmse']:.2f}, MAE={metrics['mae']:.2f}, R²={metrics['r2']:.4f}")

    # compare
    print("\n5. Comparing models...")
    compare_models(results)

    # best model
    best = min(results.keys(), key=lambda x: results[x]['rmse'])
    print(f"\nBest model: {best}")
    print(f"RMSE: {results[best]['rmse']:.2f}")
    print(f"MAE: {results[best]['mae']:.2f}")
    print(f"R²: {results[best]['r2']:.4f}")


if __name__ == "__main__":
    main()
