from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt


def calculate_rmse(y_true, y_pred):
    """
    Calculate Root Mean Square Error

    TODO: Implement this function
    """
    mse = mean_squared_error(y_true, y_pred)
    rmse = mse ** 0.5
    return rmse


def calculate_mae(y_true, y_pred):
    """
    Calculate Mean Absolute Error

    TODO: Implement this function
    """
    mae = mean_absolute_error(y_true, y_pred)
    return mae


def calculate_r2(y_true, y_pred):
    """
    Calculate R² Score

    TODO: Implement this function
    """
    r2 = r2_score(y_true, y_pred)
    return r2


def evaluate_model(model, X_test, y_test):
    """
    Evaluate a trained model and return all metrics

    TODO: Implement this function
    Should return: {'rmse': value, 'mae': value, 'r2': value}
    """
    y_pred = model.predict(X_test)
    
    rmse = calculate_rmse(y_test, y_pred)
    mae = calculate_mae(y_test, y_pred)
    r2 = calculate_r2(y_test, y_pred)
    
    return {'rmse': rmse, 'mae': mae, 'r2': r2}


def compare_models(results_dict):
    """
    Compare multiple models with visualization

    TODO: Implement this function
    Input format: {'Model Name': {'rmse': x, 'mae': y, 'r2': z}}
    """
    names = list(results_dict.keys())
    rmse_vals = [results_dict[n]['rmse'] for n in names]
    mae_vals = [results_dict[n]['mae'] for n in names]
    r2_vals = [results_dict[n]['r2'] for n in names]
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].bar(names, rmse_vals, color='skyblue')
    axes[0].set_title('RMSE')
    axes[0].set_ylabel('RMSE')
    axes[0].tick_params(axis='x', rotation=45)
    
    axes[1].bar(names, mae_vals, color='lightcoral')
    axes[1].set_title('MAE')
    axes[1].set_ylabel('MAE')
    axes[1].tick_params(axis='x', rotation=45)
    
    axes[2].bar(names, r2_vals, color='lightgreen')
    axes[2].set_title('R² Score')
    axes[2].set_ylabel('R²')
    axes[2].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig('model_comparison.png')
    plt.show()
    
    print("\n" + "="*70)
    print(f"{'Model':<20} {'RMSE':<15} {'MAE':<15} {'R²':<10}")
    print("="*70)
    for name in names:
        print(f"{name:<20} {results_dict[name]['rmse']:<15.2f} {results_dict[name]['mae']:<15.2f} {results_dict[name]['r2']:<10.4f}")
    print("="*70)
