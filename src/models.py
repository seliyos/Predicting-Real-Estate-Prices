from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
import numpy as np


# simple baseline - just predict median
def train_baseline_model(X_train, y_train):
    median_price = np.median(y_train)
    
    class MedianPredictor:
        def __init__(self, median):
            self.median = median
        
        def predict(self, X):
            return np.full(len(X), self.median)
    
    return MedianPredictor(median_price)


# basic linear regression
def train_linear_regression(X_train, y_train):
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model


# ridge with regularization
def train_ridge_regression(X_train, y_train, alpha=1.0):
    model = Ridge(alpha=alpha)
    model.fit(X_train, y_train)
    return model


# lasso 
def train_lasso_regression(X_train, y_train, alpha=1.0):
    model = Lasso(alpha=alpha)
    model.fit(X_train, y_train)
    return model


# decision tree
def train_decision_tree(X_train, y_train, max_depth=10):
    model = DecisionTreeRegressor(max_depth=max_depth, random_state=42)
    model.fit(X_train, y_train)
    return model


# random forest - usually works pretty well
def train_random_forest(X_train, y_train, n_estimators=100):
    model = RandomForestRegressor(n_estimators=n_estimators, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    return model


# gradient boosting - often the best performer
def train_gradient_boosting(X_train, y_train, n_estimators=100):
    model = GradientBoostingRegressor(n_estimators=n_estimators, random_state=42)
    model.fit(X_train, y_train)
    return model


# train everything at once
def train_all_models(X_train, y_train):
    models = {}
    
    print("Training baseline...")
    models['Baseline'] = train_baseline_model(X_train, y_train)
    
    print("Training linear regression...")
    models['Linear'] = train_linear_regression(X_train, y_train)
    
    print("Training ridge...")
    models['Ridge'] = train_ridge_regression(X_train, y_train)
    
    print("Training lasso...")
    models['Lasso'] = train_lasso_regression(X_train, y_train)
    
    print("Training decision tree...")
    models['Decision Tree'] = train_decision_tree(X_train, y_train)
    
    print("Training random forest...")
    models['Random Forest'] = train_random_forest(X_train, y_train)
    
    print("Training gradient boosting...")
    models['Gradient Boosting'] = train_gradient_boosting(X_train, y_train)
    
    return models
