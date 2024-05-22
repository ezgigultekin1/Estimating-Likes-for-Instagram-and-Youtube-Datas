import pymongo
import pandas as pd
#from config import CONNECTION_STRING, DB_NAME

class DataFetcher:
    def __init__(self, connection_string, db_name):
        self.client = pymongo.MongoClient(connection_string)
        self.db = self.client[db_name]

    def fetch_instagram_data(self, limit=5000):
        collection = self.db['instagram_media']
        data = list(collection.find().limit(limit))
        return pd.DataFrame(data)

    def fetch_youtube_data(self, limit=5000):
        collection = self.db['youtube_videos']
        data = list(collection.find().limit(limit))
        return pd.DataFrame(data)

    def fetch_influencers_data(self, limit=5000):
        collection = self.db['influencers']
        data = list(collection.find().limit(limit))
        return pd.DataFrame(data)
