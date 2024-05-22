#import pandas as pd
import numpy as np

class DataPreprocessor:
    def __init__(self):
        pass

    def clean_instagram_data(self, df):
        df.drop(columns=['_id', 'external_id', 'user_id', 'instagram_id', 'legacy_id', 'link', 'caption', 'image'], errors='ignore', inplace=True)
        df.dropna(subset=['likes_count'], inplace=True)
        df['likes_count_log'] = np.log1p(df['likes_count'])
        return df

    def clean_youtube_data(self, df):
        df.drop(columns=['_id', 'external_id', 'user_id', 'link', 'description', 'image'], errors='ignore', inplace=True)
        df.dropna(subset=['like_count'], inplace=True)
        df['like_count_log'] = np.log1p(df['like_count'])
        return df

    def remove_outliers(self, df, feature):
        Q1 = df[feature].quantile(0.25)
        Q3 = df[feature].quantile(0.75)
        IQR = Q3 - Q1
        filter = (df[feature] >= Q1 - 1.5 * IQR) & (df[feature] <= Q3 + 1.5 * IQR)
        return df.loc[filter]

    def process_dates(self, df):
        date_cols = df.select_dtypes(include=['datetime64']).columns
        for col in date_cols:
            df[col + '_year'] = df[col].dt.year
            df[col + '_month'] = df[col].dt.month
            df[col + '_day'] = df[col].dt.day
        df.drop(columns=date_cols, inplace=True)
        return df
