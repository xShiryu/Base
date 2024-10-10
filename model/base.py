import cornac
import pandas as pd
import numpy as np
import itertools
from cornac.eval_methods import RatioSplit
from cornac.models import MF, NMF, BPR, UserKNN, ItemKNN
from cornac.metrics import MAE, RMSE, Precision, Recall, NDCG, AUC, MAP
from data_load import load_and_split_data

# load the built-in MovieLens 100K and split the data based on ratio
ratings, X_train, X_test, y_train, y_test, user_encoder, movie_encoder, count = load_and_split_data()
ratings = ratings.drop(labels='timestamp', axis=1)


ml_100k = [(int(row.user_id), int(row.item_id), float(row.rating)) for row in ratings.itertuples(index=False)]
dataset = cornac.data.Dataset.from_uir(ratings.itertuples(index=False))
R = dataset.matrix.A
R_mask = (R > 0).astype(float)

# evaluation methods
#pop = cornac.eval_methods.CrossValidation(ml_100k, n_folds=5, seed=42)
rs = RatioSplit(data=ml_100k, test_size=0.2, rating_threshold=4.0, seed=42)

# initialize models
#popularity = cornac.models.MostPop()

uknn_cosine = UserKNN(name="UserKNN with cosine", k=20, similarity="cosine", verbose=False)
uknn_pearson = UserKNN(name="UserKNN with pearson", k=20, similarity="pearson", verbose=False)
uknn_cosine_idf = UserKNN(name="UserKNN with cosine/idf", k=20, similarity="cosine", weighting="idf", verbose=False)
uknn_cosine_bm25 = UserKNN(name="UserKNN with cosine/bm25", k=20, similarity="cosine", weighting="bm25", verbose=False)
uknn_pearson_idf = UserKNN(name="UserKNN with pearson/idf", k=20, similarity="pearson", weighting="idf", verbose=False)
uknn_pearson_bm25 = UserKNN(name="UserKNN with pearson/bm25", k=20, similarity="pearson", weighting="bm25", verbose=False)
uknn_pearson_amp1 = UserKNN(name="UserKNN with pearson/amp=5.0", k=20, similarity="pearson", amplify=5.0, verbose=False)
uknn_pearson_amp2 = UserKNN(name="UserKNN with pearson/amp=0.5", k=20, similarity="pearson", amplify=0.5, verbose=False)
cfu_models = [uknn_cosine, uknn_pearson, uknn_cosine_idf, uknn_cosine_bm25, uknn_pearson_idf, uknn_pearson_bm25, uknn_pearson_amp1, uknn_pearson_amp2]

iknn_cosine = ItemKNN(name="ItemKNN with cosine", k=20, similarity="cosine", verbose=False)
iknn_adjusted_cosine = ItemKNN(name="ItemKNN with adjusted cosine", k=20, similarity="cosine", mean_centered=True, verbose=False)
iknn_pearson = ItemKNN(name="ItemKNN with pearson", k=20, similarity="pearson", verbose=False)
iknn_pearson_mean = ItemKNN(name="ItemKnn with pearson/mean-centered", k=20, similarity="pearson", mean_centered=True, verbose=False)
cfi_models = [iknn_cosine, iknn_adjusted_cosine, iknn_pearson, iknn_pearson_mean]

mf = MF(name="MF (k=10)", k=10, max_iter=100, learning_rate=0.01, lambda_reg=0.0, use_bias=False, verbose=False)
mf_iter = MF(name="MF (max_iter=20)", k=10, max_iter=20, learning_rate=0.01, lambda_reg=0.0, use_bias=False, verbose=False)
mf_lambda = MF(name="MF (lambda_reg=0.01)", k=10, max_iter=20, learning_rate=0.01, lambda_reg=0.01, use_bias=False, verbose=False)
mf_bias = MF(name="MF (bias)", k=10, max_iter=20, learning_rate=0.01, lambda_reg=0.01, use_bias=True, verbose=False)

nmf = NMF(name="NMF (k=10)", k=10, max_iter=100, learning_rate=0.01, lambda_reg=0.0, use_bias=False, verbose=False)
nmf_iter = NMF(name="NMF (max_iter=20)", k=10, max_iter=20, learning_rate=0.01, lambda_reg=0.0, use_bias=False, verbose=False)
nmf_lambda = NMF(name="NMF (lambda_reg=0.01)", k=10, max_iter=20, learning_rate=0.01, lambda_reg=0.01, use_bias=False, verbose=False)
nmf_bias = NMF(name="NMF (bias)", k=10, max_iter=20, learning_rate=0.01, lambda_reg=0.01, use_bias=True, verbose=False)
mf_models = [mf, mf_iter, mf_lambda, mf_bias, nmf, nmf_iter, nmf_lambda, nmf_bias]

# define metrics to evaluate the models
metrics = [MAE(), RMSE(), Recall(k=10), NDCG(k=10), Precision(k=10)]

# put it together in an experiment
#print("Phương pháp gợi ý dựa trên thống kê độ phổ biến:")
#cornac.Experiment(eval_method=pop, models=[popularity], metrics=metrics).run()

print("Phương pháp gợi ý dựa trên lọc cộng tác (hướng người dùng):")
cornac.Experiment(eval_method=rs, models=cfu_models, metrics=metrics, user_based=True).run()

print("Phương pháp gợi ý dựa trên lọc cộng tác (hướng mục):")
cornac.Experiment(eval_method=rs, models=cfi_models, metrics=metrics, user_based=True).run()

print("Phương pháp gợi ý dựa trên ma trận nhân tử:")
cornac.Experiment(eval_method=rs, models=mf_models, metrics=metrics).run()