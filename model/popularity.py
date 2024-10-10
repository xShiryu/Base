import pandas as pd
import numpy as np
import time
from sklearn.metrics import mean_squared_error
from data_load import load_and_split_data

### Load and split dataset
ratings, X_train, X_test, y_train, y_test, user_encoder, movie_encoder, count = load_and_split_data()
train_data = pd.concat([X_train, y_train], axis=1)
test_data = pd.concat([X_test, y_test], axis=1)

### Filter movies with more reviews
m = count['vote_count'].quantile(0.9)
C = count['avg_score'].mean()

popular_titles = count.copy().loc[count['vote_count'] >= m]

### Calculate imdb score to evaluate popularity
def imdb_weighted_ratings(count, m = m, C = C):
    v = count['vote_count']
    R = count['avg_score']
    return ((v*R)/(v+m)) + ((m*C)/(v+m))

popular_titles["imdb_score"] = popular_titles.apply(imdb_weighted_ratings, axis = 1)
popular_titles = popular_titles.sort_values('imdb_score', ascending=False)


def recommend_items(user_id, ratings, popular_titles, num_recommendation = 5):
    rated_movies = ratings[ratings['user_id'] == user_id]['item_id'].unique()

    recommendations = popular_titles[~popular_titles['item_id'].isin(rated_movies)]

    return recommendations.sort_values(by='imdb_score', ascending=False).head(num_recommendation)
    
    #return popular_titles[:num_recommendation]

def evaluate_rmse(train_data, test_data, recommend_func, popular_titles):
    y_pred = []
    y_true = []

    for user_id in test_data['user_id'].unique():
        # Get the movies rated by the user in the test set
        true_ratings = test_data[test_data['user_id'] == user_id][['item_id', 'rating']]
        
        # Get recommended items
        recommendations = recommend_func(user_id, train_data, popular_titles, len(true_ratings))
        
        # Loop through the test items to record predicted vs actual ratings
        for _, row in true_ratings.iterrows():
            item_id = row['item_id']
            if item_id in recommendations['item_id'].values:
                y_pred.append(recommendations[recommendations['item_id'] == item_id]['avg_score'].values[0])
                y_true.append(row['rating'])
    
    return np.sqrt(mean_squared_error(y_true, y_pred))

def precision_at_k(user_id, true_items, recommendations, k=5):
    top_k_recs = recommendations.head(k)['item_id']
    true_items_set = set(true_items['item_id'].values)
    recommended_set = set(top_k_recs.values)
    
    return len(true_items_set & recommended_set) / float(k)

def recall_at_k(user_id, true_items, recommendations, k=5):
    top_k_recs = recommendations.head(k)['item_id']
    true_items_set = set(true_items['item_id'].values)
    recommended_set = set(top_k_recs.values)
    
    return len(true_items_set & recommended_set) / float(len(true_items_set))

def evaluate_precision_recall(train_data, test_data, recommend_func, popular_titles, k=5):
    precision_scores = []
    recall_scores = []
    
    for user_id in test_data['user_id'].unique():
        true_ratings = test_data[test_data['user_id'] == user_id][['item_id', 'rating']]
        
        # Recommend items for the user
        recommendations = recommend_func(user_id, train_data, popular_titles, k)
        
        # Calculate precision and recall
        precision = precision_at_k(user_id, true_ratings, recommendations, k)
        recall = recall_at_k(user_id, true_ratings, recommendations, k)
        
        precision_scores.append(precision)
        recall_scores.append(recall)
    
    avg_precision = np.mean(precision_scores)
    avg_recall = np.mean(recall_scores)
    
    return avg_precision, avg_recall

# Measure training time (loading and calculating IMDb score)
start_train_time = time.time()

precision, recall = evaluate_precision_recall(train_data, test_data, recommend_items, popular_titles, k=10)
print(f"Precision@10: {precision:.4f}")
print(f"Recall@10: {recall:.4f}")

end_train_time = time.time()
train_time = end_train_time - start_train_time

# Measure testing (evaluation) time
start_test_time = time.time()

rmse = evaluate_rmse(train_data, test_data, recommend_items, popular_titles)
print(f"RMSE: {rmse:.4f}")

end_test_time = time.time()
test_time = end_test_time - start_test_time

print(f"Training time: {train_time:.4f} seconds")
print(f"Testing time: {test_time:.4f} seconds")

user_id = test_data['user_id'].iloc[int(input())]
recommended_movies = recommend_items(user_id, train_data, popular_titles, num_recommendation=5)
print(recommended_movies[['item_id', 'vote_count', 'avg_score', "imdb_score"]])