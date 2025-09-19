from skopt import gp_minimize
from skopt.space import Real, Integer

def objective_function(params):
    """Objective function for Bayesian optimization"""
    alpha, threshold = params
    
    model = RidgeModel({'alpha': alpha})
    # Run cross-validation with threshold
    cv_score = cross_validate_with_threshold(model, X_train, y_train, threshold)
    
    return -cv_score['sharpe_ratio']  # Minimize negative Sharpe

# Define search space
space = [Real(0.001, 100.0, 'log-uniform', name='alpha'),
         Real(0.0, 0.01, name='threshold')]

# Run optimization
result = gp_minimize(objective_function, space, n_calls=50, random_state=42)