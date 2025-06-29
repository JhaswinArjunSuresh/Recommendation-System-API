import pandas as pd
from implicit.als import AlternatingLeastSquares
import scipy.sparse as sparse
import numpy as np

class RecommenderSystem:
    def __init__(self):
        df = pd.read_csv("data/interactions.csv")
        user_ids = df["user_id"].astype("category")
        item_ids = df["item_id"].astype("category")
        self.user_mapping = dict(enumerate(user_ids.cat.categories))
        self.item_mapping = dict(enumerate(item_ids.cat.categories))
        self.reverse_user_mapping = {v: k for k, v in self.user_mapping.items()}
        self.reverse_item_mapping = {v: k for k, v in self.item_mapping.items()}

        rows = user_ids.cat.codes
        cols = item_ids.cat.codes
        values = df["interaction"]

        self.user_item_matrix = sparse.coo_matrix(
            (values, (rows, cols)),
            shape=(len(user_ids.cat.categories), len(item_ids.cat.categories))
        ).tocsr()

        self.model = AlternatingLeastSquares(factors=20, regularization=0.1, iterations=15)
        self.model.fit(self.user_item_matrix)

    def recommend_for_user(self, user_id, N=5):
        user_idx = self.reverse_user_mapping.get(user_id)
        if user_idx is None:
            return []
        recs = self.model.recommend(user_idx, self.user_item_matrix, N=N)
        return [{"item_id": self.item_mapping[i], "score": float(score)} for i, score in recs]

    def similar_items(self, item_id, N=5):
        item_idx = self.reverse_item_mapping.get(item_id)
        if item_idx is None:
            return []
        sims = self.model.similar_items(item_idx, N)
        return [{"item_id": self.item_mapping[i], "score": float(score)} for i, score in sims]

