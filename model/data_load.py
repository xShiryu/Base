import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def load_and_split_data():
    # Define column names
    column_names = ['user_id', 'item_id', 'rating', 'timestamp']
    
    # Load the dataset
    path = "../data/"
    ratings = pd.read_csv(path + 'u.data', sep='\t', names=column_names)
    count = ratings.groupby('item_id').agg(vote_count = ('rating', 'size'), avg_score = ('rating', 'mean')).reset_index()
    count = count.sort_values(by=['vote_count', 'avg_score'], ascending=[False, False]).reset_index()

    #print(count)
    
    # Encode user_id and item_id
    user_encoder = LabelEncoder()
    movie_encoder = LabelEncoder()

    ratings['user_id'] = user_encoder.fit_transform(ratings['user_id'])
    ratings['item_id'] = movie_encoder.fit_transform(ratings['item_id'])
    
    # Prepare features (X) and labels (y)
    X = ratings[['user_id', 'item_id']]
    y = ratings['rating']
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Return the split data and the encoders
    return ratings, X_train, X_test, y_train, y_test, user_encoder, movie_encoder, count