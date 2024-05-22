from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import VotingRegressor

class ModelTrainer:
    def __init__(self):
        self.models = {
            'GradientBoosting': GradientBoostingRegressor(),
            'RandomForest': RandomForestRegressor(),
            'XGBoost': XGBRegressor(),
            'NeuralNetwork': MLPRegressor()
        }

    def hyperparameter_tuning(self, model_name, X_train, y_train):
        param_grid = self.get_param_grid(model_name)
        grid_search = GridSearchCV(self.models[model_name], param_grid, cv=5, scoring='neg_mean_squared_error')
        grid_search.fit(X_train, y_train)
        print(f"Best parameters for {model_name}: {grid_search.best_params_}")
        return grid_search.best_estimator_

    def get_param_grid(self, model_name):
        if model_name == 'GradientBoosting':
            return {
                'n_estimators': [100, 200],
                'learning_rate': [0.05, 0.1],
                'max_depth': [3, 5]
            }
        elif model_name == 'RandomForest':
            return {
                'n_estimators': [100, 200],
                'max_depth': [3, 5]
            }
        elif model_name == 'XGBoost':
            return {
                'n_estimators': [100, 200],
                'learning_rate': [0.05, 0.1],
                'max_depth': [3, 5]
            }
        elif model_name == 'NeuralNetwork':
            return {
                'hidden_layer_sizes': [(100,), (100, 100)],
                'alpha': [0.0001, 0.001, 0.01],
                'max_iter': [500, 1000, 2000]  # Eğitim iterasyon sayısını artırma
            }

    def evaluate_model(self, model, X_train, y_train):
        from sklearn.model_selection import cross_val_score
        scores = cross_val_score(model, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
        mse = -scores.mean()
        std = scores.std()
        return mse, std

    def ensemble_model(self, models):
        estimators = [(name, model) for name, model in models]
        ensemble = VotingRegressor(estimators=estimators)
        return ensemble
