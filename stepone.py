import pandas as pd
from pymongo import MongoClient

#Connect to MongoDB
connection_string = 'mongodb://wearisma-read-only:'
client = MongoClient(connection_string)


# Select the database and collections
db = client['heroku_62fv85nh']
instagram_media = db['instagram_media']
youtube_videos = db['youtube_videos']
influencers = db['influencers']

# Instagram and YouTube data queries
instagram_query = {'has_hidden_likes': False}
instagram_fields = {'_id': 1, 'likes_count': 1, 'user_id': 1}
youtube_query = {'like_count': {'$gt': 0}}
youtube_fields = {'_id': 1, 'like_count': 1, 'channel_id': 1}

# Get Instagram and YouTube data
instagram_data = instagram_media.find(instagram_query).limit(500000)
youtube_data = youtube_videos.find(youtube_query).limit(500000)

# Convert data to DataFrames
df_instagram = pd.DataFrame(list(instagram_data))
df_youtube = pd.DataFrame(list(youtube_data))

# Get influencer data
instagram_user_ids = df_instagram['user_id'].unique()
youtube_channel_ids = df_youtube['channel_id'].unique()

influencers_instagram_data = influencers.find({'instagram_user_id': {'$in': list(instagram_user_ids)}})
influencers_youtube_data = influencers.find({'youtube_channel_id': {'$in': list(youtube_channel_ids)}})

df_influencers_instagram = pd.DataFrame(list(influencers_instagram_data))
df_influencers_youtube = pd.DataFrame(list(influencers_youtube_data))

# Merge Instagram and YouTube data
df_merged_instagram = pd.merge(df_instagram, df_influencers_instagram,
                               left_on='user_id',
                               right_on='instagram_user_id',
                               how='left')

df_merged_youtube = pd.merge(df_youtube, df_influencers_youtube,
                             left_on='channel_id',
                             right_on='youtube_channel_id',
                             how='left')

# Remove duplicate records and unnecessary columns
df_merged_instagram = df_merged_instagram.drop_duplicates(subset='user_id').dropna(axis=1, how='all')
df_merged_youtube = df_merged_youtube.drop_duplicates(subset='channel_id').dropna(axis=1, how='all')

# Rename column names (_id_x to _id)
df_merged_instagram.rename(columns={'_id_x': '_id'}, inplace=True)
df_merged_youtube.rename(columns={'_id_x': '_id'}, inplace=True)

# Print the results
print("Instagram Data Sample:")
print(df_merged_instagram)  # Prints the entire DataFrame
print("\nYouTube Data Sample:")
print(df_merged_youtube)  # Prints the entire DataFrame
