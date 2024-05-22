import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from data_fetcher import DataFetcher
from data_preprocessor import DataPreprocessor
from model_training import ModelTrainer
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.metrics import mean_squared_error, r2_score
from config import CONNECTION_STRING, DB_NAME
from sklearn.preprocessing import StandardScaler
import pickle
import os

# Change the backend to 'Agg' for matplotlib to avoid tkinter issues
plt.switch_backend('Agg')

def enhance_features(df, target):
    df = impute_missing_values(df)
    df = create_interaction_terms(df)
    df = create_advanced_features(df)
    df[target + '_log'] = np.log1p(df[target])
    return df

def impute_missing_values(df):
    from sklearn.impute import SimpleImputer
    imputer = SimpleImputer(strategy='median')
    numeric_df = df.select_dtypes(include=[np.number])
    df_imputed = pd.DataFrame(imputer.fit_transform(numeric_df), columns=numeric_df.columns)
    df[numeric_df.columns] = df_imputed
    return df

def create_interaction_terms(df):
    from sklearn.preprocessing import PolynomialFeatures
    poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
    numeric_df = df.select_dtypes(include=[np.number])
    # Impute again to make sure no NaNs are present
    numeric_df = numeric_df.fillna(numeric_df.median())
    interaction_terms = poly.fit_transform(numeric_df)
    interaction_df = pd.DataFrame(interaction_terms, columns=poly.get_feature_names_out(numeric_df.columns))
    df = df.join(interaction_df, rsuffix='_interaction')
    return df

def create_advanced_features(df):
    if 'description' in df.columns:
        df['description_length'] = df['description'].apply(lambda x: len(str(x)))
    return df

def preprocess_data(df):
    df = df.select_dtypes(include=[np.number])
    scaler = StandardScaler()
    df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
    return df_scaled

def plot_learning_curve(estimator, X, y, title, filename):
    train_sizes, train_scores, test_scores = learning_curve(estimator, X, y, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
    train_scores_mean = -train_scores.mean(axis=1)
    train_scores_std = train_scores.std(axis=1)
    test_scores_mean = -test_scores.mean(axis=1)
    test_scores_std = test_scores.std(axis=1)

    plt.figure()
    plt.title(title)
    plt.xlabel("Training size")
    plt.ylabel("MSE")
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training error")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation error")

    plt.legend(loc="best")
    plt.savefig(filename)
    plt.close()

def main():
    fetcher = DataFetcher(CONNECTION_STRING, DB_NAME)
    preprocessor = DataPreprocessor()
    trainer = ModelTrainer()

    print("Fetching Instagram data...")
    instagram_data = fetcher.fetch_instagram_data()
    print("Instagram data fetched successfully.")
    print("Processing Instagram data...")
    instagram_data = preprocessor.clean_instagram_data(instagram_data)
    instagram_data = preprocessor.process_dates(instagram_data)
    instagram_data = enhance_features(instagram_data, 'likes_count')
    instagram_data = preprocessor.remove_outliers(instagram_data, 'likes_count_log')
    X_insta = preprocess_data(instagram_data.drop(columns=['likes_count', 'likes_count_log']))
    y_insta = instagram_data['likes_count_log']
    X_insta_train, X_insta_test, y_insta_train, y_insta_test = train_test_split(X_insta, y_insta, test_size=0.2, random_state=42)

    best_models = []
    for model_name in trainer.models.keys():
        best_model = trainer.hyperparameter_tuning(model_name, X_insta_train, y_insta_train)
        best_models.append((model_name, best_model))
        mse, std = trainer.evaluate_model(best_model, X_insta_train, y_insta_train)
        print(f"Evaluating {model_name} model for Instagram...")
        print(f"{model_name} MSE: {mse}, STD: {std}")
        plot_learning_curve(best_model, X_insta_train, y_insta_train,
                            f"Learning curve for {model_name} on Instagram data",
                            f"learning_curve_{model_name}_instagram.png")

    ensemble_model = trainer.ensemble_model(best_models)
    print("Evaluating Ensemble model for Instagram...")
    mse, std = trainer.evaluate_model(ensemble_model, X_insta_train, y_insta_train)
    print(f"Ensemble MSE: {mse}, STD: {std}")

    best_model = min(best_models, key=lambda x: trainer.evaluate_model(x[1], X_insta_train, y_insta_train)[0])[1]
    best_model.fit(X_insta_train, y_insta_train)
    y_insta_pred = best_model.predict(X_insta_test)
    mse = mean_squared_error(y_insta_test, y_insta_pred)
    r2 = r2_score(y_insta_test, y_insta_pred)
    print(f"Evaluating best model for Instagram on test data...")
    print(f"MSE: {mse}")
    print(f"R^2: {r2}")

    print("Saving Instagram model pipeline...")
    with open("instagram_model_pipeline.pkl", "wb") as f:
        pickle.dump(best_model, f)

    print("Fetching YouTube data...")
    youtube_data = fetcher.fetch_youtube_data()
    print("YouTube data fetched successfully.")
    print("Processing YouTube data...")
    youtube_data = preprocessor.clean_youtube_data(youtube_data)
    youtube_data = preprocessor.process_dates(youtube_data)
    youtube_data = enhance_features(youtube_data, 'like_count')
    youtube_data = preprocessor.remove_outliers(youtube_data, 'like_count_log')
    X_youtube = preprocess_data(youtube_data.drop(columns=['like_count', 'like_count_log']))
    y_youtube = youtube_data['like_count_log']
    X_youtube_train, X_youtube_test, y_youtube_train, y_youtube_test = train_test_split(X_youtube, y_youtube, test_size=0.2, random_state=42)

    best_models = []
    for model_name in trainer.models.keys():
        best_model = trainer.hyperparameter_tuning(model_name, X_youtube_train, y_youtube_train)
        best_models.append((model_name, best_model))
        mse, std = trainer.evaluate_model(best_model, X_youtube_train, y_youtube_train)
        print(f"Evaluating {model_name} model for YouTube...")
        print(f"{model_name} MSE: {mse}, STD: {std}")
        plot_learning_curve(best_model, X_youtube_train, y_youtube_train,
                            f"Learning curve for {model_name} on YouTube data",
                            f"learning_curve_{model_name}_youtube.png")

    ensemble_model = trainer.ensemble_model(best_models)
    print("Evaluating Ensemble model for YouTube...")
    mse, std = trainer.evaluate_model(ensemble_model, X_youtube_train, y_youtube_train)
    print(f"Ensemble MSE: {mse}, STD: {std}")

    best_model = min(best_models, key=lambda x: trainer.evaluate_model(x[1], X_youtube_train, y_youtube_train)[0])[1]
    best_model.fit(X_youtube_train, y_youtube_train)
    y_youtube_pred = best_model.predict(X_youtube_test)
    mse = mean_squared_error(y_youtube_test, y_youtube_pred)
    r2 = r2_score(y_youtube_test, y_youtube_pred)
    print(f"Evaluating best model for YouTube on test data...")
    print(f"MSE: {mse}")
    print(f"R^2: {r2}")

    print("Saving YouTube model pipeline...")
    with open("youtube_model_pipeline.pkl", "wb") as f:
        pickle.dump(best_model, f)

if __name__ == "__main__":
    main()
