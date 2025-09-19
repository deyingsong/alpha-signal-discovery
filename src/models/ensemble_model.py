class StackedEnsemble:
    def __init__(self):
        self.base_models = {
            'ridge': RidgeModel(config),
            'rf': RandomForestModel(config), 
            'xgb': XGBoostModel(config),
            'lgb': LightGBMModel(config)  # Add LightGBM
        }
        self.meta_model = Ridge(alpha=1.0)
    
    def fit(self, X, y):
        # Train base models with cross-validation
        meta_features = np.zeros((len(X), len(self.base_models)))
        
        for i, (name, model) in enumerate(self.base_models.items()):
            cv_predictions = cross_val_predict(model, X, y, cv=TimeSeriesSplit(5))
            meta_features[:, i] = cv_predictions
            model.fit(X, y)  # Refit on full data
        
        # Train meta-model
        self.meta_model.fit(meta_features, y)