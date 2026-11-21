"""Step 3: popularity, TF-IDF content, and user-based collaborative hybrid.

All methods return ranked dictionaries, normally Top-5.  This implementation
uses only the ratings passed to ``fit``; Stage 4 can fit it on a train split
without leaking hidden test ratings into the collaborative or sentiment scores.

Algorithm summary
-----------------
Popularity: Bayesian-smoothed rating average, with a minimum number of ratings.
Content: cosine similarity against TF-IDF from ``GameNLPPipeline``.  Search by
    free text, a known item, or a user's positively/negatively weighted profile.
Hybrid: weighted mixture of user-based collaborative filtering (CF), content
    similarity, and lexicon review sentiment. Missing evidence (e.g. CF on a
    new game) is *not* treated as a negative vote; available weights are
    normalized.  This makes unrated cold-start games recommendable by content.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity

from .nlp_pipeline import GameNLPPipeline, REQUIRED_RATING_COLUMNS


class GameRecommender:
    """Small, explainable recommender that integrates the Step 2 NLP pipeline."""

    def __init__(
        self,
        content_weight: float = 0.50,
        cf_weight: float = 0.35,
        sentiment_weight: float = 0.15,
        min_common_ratings: int = 2,
        neighbor_shrinkage: float = 3.0,
    ) -> None:
        weights = (content_weight, cf_weight, sentiment_weight)
        if any(w < 0 for w in weights) or sum(weights) <= 0:
            raise ValueError("Hybrid weights must be nonnegative and not all zero")
        if min_common_ratings < 1 or neighbor_shrinkage < 0:
            raise ValueError("Invalid collaborative-filtering neighbor settings")
        self.weights = {
            "content": float(content_weight),
            "collaborative": float(cf_weight),
            "sentiment": float(sentiment_weight),
        }
        self.min_common_ratings = int(min_common_ratings)
        self.neighbor_shrinkage = float(neighbor_shrinkage)
        self.nlp = GameNLPPipeline()
        self.items: pd.DataFrame | None = None
        self.ratings: pd.DataFrame | None = None
        self.users: dict[str, dict[str, float]] = {}
        self.user_means: dict[str, float] = {}
        self.sentiment_scores: dict[str, float] = {}
        self.global_mean = 3.0

    def fit(self, items: pd.DataFrame, ratings: pd.DataFrame) -> "GameRecommender":
        """Fit from games and *training-only* reviews/ratings.

        Each (user_id, item_id) must occur once, ratings must be in [1, 5],
        and all rated item IDs must exist in the items table.
        """
        missing = set(REQUIRED_RATING_COLUMNS) - set(ratings.columns)
        if missing:
            raise ValueError(f"Missing rating columns: {sorted(missing)}")
        ratings = ratings.copy()
        if (ratings[["user_id", "item_id"]].isna().any().any()
                or ratings.duplicated(["user_id", "item_id"]).any()):
            raise ValueError("Missing user/item IDs or duplicate user-item ratings")
        numeric = pd.to_numeric(ratings["rating"], errors="coerce")
        if numeric.isna().any() or not numeric.between(1, 5).all():
            raise ValueError("Ratings must be numeric and between 1 and 5")
        ratings["rating"] = numeric.astype(float)
        ratings["user_id"] = ratings["user_id"].astype(str)
        ratings["item_id"] = ratings["item_id"].astype(str)
        if not set(ratings["item_id"]).issubset(set(items["item_id"].astype(str))):
            raise ValueError("ratings.csv refers to unknown game item IDs")
        self.nlp.fit(items, ratings if not ratings.empty else None)
        self.items = self.nlp.items
        self.ratings = ratings.reset_index(drop=True)
        self.users = {
            str(user_id): dict(zip(group["item_id"], group["rating"]))
            for user_id, group in ratings.groupby("user_id")
        }
        self.user_means = {
            user_id: float(np.mean(list(history.values())))
            for user_id, history in self.users.items()
        }
        self.global_mean = float(ratings["rating"].mean()) if not ratings.empty else 3.0
        if ratings.empty:
            self.sentiment_scores = {}
        else:
            sentiment_summary = self.nlp.item_review_sentiment()
            self.sentiment_scores = dict(zip(
                sentiment_summary["item_id"].astype(str),
                sentiment_summary["mean_sentiment"].astype(float),
            ))
        return self

    def _require_fit(self) -> None:
        if self.items is None or self.ratings is None:
            raise RuntimeError("Call fit(items, ratings) first")

    def _user_history(self, user_id: str) -> dict[str, float]:
        self._require_fit()
        user_id = str(user_id)
        if user_id not in self.users:
            raise KeyError(f"Unknown user_id: {user_id}")
        return self.users[user_id]

    def _result(self, index: int, *, score: float, method: str,
                keywords: list[str] | None = None, **extra: Any) -> dict[str, Any]:
        assert self.items is not None
        game = self.items.iloc[index]
        return {
            "item_id": str(game["item_id"]),
            "title": str(game["title"]),
            "method": method,
            "score": round(float(score), 4),
            "keywords": keywords or [],
            **extra,
        }

    def recommend_popularity(
        self, user_id: str | None = None, top_k: int = 5,
        min_rating_count: int = 3, prior_count: float = 3.0,
    ) -> list[dict[str, Any]]:
        """Rank well-rated games after a minimum rating count threshold.

        Bayesian prior discourages two lucky five-star ratings from always
        outranking well-established games:
          score = (sum_of_ratings + prior_count*global_mean) / (n + prior_count)
        """
        self._require_fit()
        if top_k < 1:
            return []
        if min_rating_count < 1 or prior_count < 0:
            raise ValueError("min_rating_count >= 1 and prior_count >= 0 required")
        seen = set(self._user_history(user_id)) if user_id is not None else set()
        assert self.ratings is not None and self.items is not None
        stats = self.ratings.groupby("item_id")["rating"].agg(["count", "sum", "mean"])
        ranked = []
        for i, game in self.items.iterrows():
            game_id = str(game["item_id"])
            if game_id in seen or game_id not in stats.index:
                continue
            info = stats.loc[game_id]
            n = int(info["count"])
            if n < min_rating_count:
                continue
            score = (float(info["sum"]) + prior_count * self.global_mean) / (n + prior_count)
            ranked.append(self._result(
                i, score=score, method="popularity", rating_count=n,
                average_rating=round(float(info["mean"]), 4),
                keywords=str(game["genre_tags"]).split("|")[:3],
            ))
        ranked.sort(key=lambda r: (-r["score"], -r["rating_count"], r["item_id"]))
        return ranked[:top_k]

    def _profile_vector(self, user_id: str) -> csr_matrix | None:
        """Preference profile weighted by (rating - 3): liked +, disliked -."""
        history = self._user_history(user_id)
        assert self.nlp.tfidf_matrix is not None
        ids = [item_id for item_id in history if item_id in self.nlp.item_index]
        if not ids:
            return None
        idx = [self.nlp.item_index[item_id] for item_id in ids]
        weights = np.asarray([history[item_id] - 3.0 for item_id in ids])
        if not np.any(weights):  # No directional preference to model.
            return None
        profile = csr_matrix(weights.reshape(1, -1)) @ self.nlp.tfidf_matrix[idx]
        magnitude = np.sqrt(profile.multiply(profile).sum())
        return (profile / magnitude).tocsr() if magnitude > 1e-12 else None

    def _content_scores(
        self, user_id: str,
    ) -> tuple[np.ndarray, csr_matrix | None]:
        """Cosine similarities for every game against user's TF-IDF profile."""
        self._require_fit()
        assert self.items is not None and self.nlp.tfidf_matrix is not None
        profile = self._profile_vector(user_id)
        if profile is None:
            return np.zeros(len(self.items)), None
        raw = cosine_similarity(profile, self.nlp.tfidf_matrix).ravel()
        return np.clip(raw, 0.0, 1.0), profile

    def recommend_content(
        self, *, query: str | None = None, user_id: str | None = None,
        item_id: str | None = None, top_k: int = 5,
    ) -> list[dict[str, Any]]:
        """Top-K by TF-IDF cosine similarity: query OR user OR existing item.

        For player profiles, the results exclude every game the player rated.
        For item-to-item, the source game is excluded.
        """
        self._require_fit()
        if sum(value is not None for value in (query, user_id, item_id)) != 1:
            raise ValueError("Pass exactly one: query=, user_id=, or item_id=")
        if top_k < 1:
            return []
        if query is not None:
            raw = self.nlp.search(query, top_k)
            return [
                {"item_id": hit["item_id"], "title": hit["title"],
                 "method": "content", "score": hit["similarity"],
                 "similarity": hit["similarity"], "keywords": hit["keywords"]}
                for hit in raw
            ]
        if item_id is not None:
            raw = self.nlp.similar_games(str(item_id), top_k)
            return [
                {"item_id": hit["item_id"], "title": hit["title"],
                 "method": "content", "score": hit["similarity"],
                 "similarity": hit["similarity"], "keywords": hit["keywords"]}
                for hit in raw
            ]
        assert user_id is not None and self.items is not None
        scores, profile = self._content_scores(user_id)
        if profile is None:
            return []
        seen = set(self._user_history(user_id))
        ranked = sorted(
            (i for i in range(len(scores))
             if str(self.items.iloc[i]["item_id"]) not in seen and scores[i] > 0),
            key=lambda i: (-scores[i], str(self.items.iloc[i]["item_id"])),
        )[:top_k]
        # Keyword attribution only from positive profile terms.
        positive_profile = profile.maximum(0)
        return [self._result(
            i, score=scores[i], method="content", similarity=round(float(scores[i]), 4),
            keywords=self.nlp.matching_keywords(positive_profile, i),
        ) for i in ranked]

    def _neighbor_similarity(self, user_a: str, user_b: str) -> float:
        """Mean-centered cosine on co-rated games, shrunk for small overlaps."""
        a, b = self.users[user_a], self.users[user_b]
        overlap = sorted(a.keys() & b.keys())
        if len(overlap) < self.min_common_ratings:
            return 0.0
        va = np.asarray([a[k] - self.user_means[user_a] for k in overlap])
        vb = np.asarray([b[k] - self.user_means[user_b] for k in overlap])
        denom = np.linalg.norm(va) * np.linalg.norm(vb)
        if denom < 1e-12:
            return 0.0
        similarity = float(np.dot(va, vb) / denom)
        # Only trust positively correlated neighbors for this small dataset.
        return max(0.0, similarity) * len(overlap) / (
            len(overlap) + self.neighbor_shrinkage
        )

    def collaborative_predictions(self, user_id: str) -> dict[str, float]:
        """Predict unseen game ratings via positively similar neighboring users.

        r_hat(u,i) = mean_u + sum_v(sim(u,v) * (r(v,i)-mean_v)) / sum_v(sim(u,v))

        Return only games with supporting neighbor observations. Predictions
        are clipped to the legal 1–5 rating interval.
        """
        self._require_fit()
        history = self._user_history(user_id)
        assert self.items is not None
        neighbors = {
            other: self._neighbor_similarity(str(user_id), other)
            for other in self.users if other != str(user_id)
        }
        neighbors = {u: sim for u, sim in neighbors.items() if sim > 0}
        estimates = {}
        for game_id in self.items["item_id"].astype(str):
            if game_id in history:
                continue
            observations = [(sim, self.users[u][game_id] - self.user_means[u])
                            for u, sim in neighbors.items() if game_id in self.users[u]]
            if not observations:
                continue
            numerator = sum(sim * deviation for sim, deviation in observations)
            denominator = sum(sim for sim, _ in observations)
            estimates[game_id] = round(float(np.clip(
                self.user_means[str(user_id)] + numerator / denominator, 1.0, 5.0
            )), 4)
        return estimates

    def recommend_hybrid(self, user_id: str, top_k: int = 5) -> list[dict[str, Any]]:
        """Combine content, user-user CF, and review sentiment for unseen games.

        Normalizations: content cosine in [0, 1]; CF prediction (r - 1)/4;
        sentiment (mean_review_score + 1)/2, only if reviews exist.
        The weighted mean is calculated only from available signals. In
        particular, an unrated cold-start game is eligible on content alone.
        """
        self._require_fit()
        if top_k < 1:
            return []
        seen = set(self._user_history(user_id))
        assert self.items is not None
        content_scores, profile = self._content_scores(user_id)
        cf_scores = self.collaborative_predictions(user_id)
        ranked: list[dict[str, Any]] = []
        for i, game in self.items.iterrows():
            game_id = str(game["item_id"])
            if game_id in seen:
                continue
            components: dict[str, float] = {}
            if profile is not None and content_scores[i] > 0:
                components["content"] = float(content_scores[i])
            if game_id in cf_scores:
                components["collaborative"] = (cf_scores[game_id] - 1) / 4
            if game_id in self.sentiment_scores:
                components["sentiment"] = float(np.clip(
                    (self.sentiment_scores[game_id] + 1) / 2, 0, 1
                ))
            # Sentiment alone says whether other players liked a game, not
            # whether *this user* would. Require a personalized signal.
            if not ({"content", "collaborative"} & components.keys()):
                continue
            active_weight = sum(self.weights[name] for name in components)
            if active_weight <= 0:
                continue
            final = sum(self.weights[name] * value for name, value in components.items()) / active_weight
            keywords = (self.nlp.matching_keywords(profile.maximum(0), i)
                        if profile is not None and "content" in components else [])
            ranked.append(self._result(
                i, score=final, method="hybrid", keywords=keywords,
                content_similarity=round(float(content_scores[i]), 4),
                predicted_rating=cf_scores.get(game_id),
                review_sentiment=self.sentiment_scores.get(game_id),
                components={key: round(val, 4) for key, val in components.items()},
            ))
        ranked.sort(key=lambda hit: (-hit["score"], hit["item_id"]))
        return ranked[:top_k]

    def add_cold_start_game(
        self, *, item_id: str, title: str, genre_tags: str,
        store_description: str, platform: str = "PC",
        play_mode: str = "single-player",
    ) -> "GameRecommender":
        """Register a new game using metadata only (no ratings or reviews).

        Refit TF-IDF so the new description is included in the vocabulary. The
        *existing* training interactions stay unchanged. A new game cannot be
        ranked by popularity or CF yet, but can be found by text similarity.
        """
        self._require_fit()
        assert self.items is not None and self.ratings is not None
        if not str(item_id).strip() or not str(title).strip():
            raise ValueError("The cold-start game needs item_id and title")
        if str(item_id) in self.nlp.item_index:
            raise ValueError(f"Duplicate item_id: {item_id}")
        new_game = pd.DataFrame([{
            "item_id": str(item_id), "title": str(title), "platform": platform,
            "genre_tags": genre_tags, "store_description": store_description,
            "play_mode": play_mode,
        }])
        return self.fit(pd.concat([self.items, new_game], ignore_index=True), self.ratings)


def print_recommendations(matches: list[dict[str, Any]]) -> None:
    """Print ranking and explanations for terminal demonstrations."""
    if not matches:
        print("  No recommendations available for the chosen input.")
    for rank, hit in enumerate(matches, 1):
        extra = []
        if "predicted_rating" in hit and hit["predicted_rating"] is not None:
            extra.append(f"CF estimate={hit['predicted_rating']:.2f}/5")
        if "rating_count" in hit:
            extra.append(f"{hit['rating_count']} ratings")
        if hit["keywords"]:
            extra.append("keywords=" + ", ".join(hit["keywords"]))
        print(f"  {rank}. {hit['title']} [{hit['item_id']}] "
              f"| {hit['method']} score={hit['score']:.4f}"
              + (" | " + " | ".join(extra) if extra else ""))


def main() -> None:
    """Run Step 3 with the assignment's game query and personalized examples."""
    root = Path(__file__).resolve().parents[1]
    items = pd.read_csv(root / "data" / "items.csv")
    ratings = pd.read_csv(root / "data" / "ratings.csv")
    recommender = GameRecommender().fit(items, ratings)
    query = "story driven open world exploration with crafting"
    user_id = "U01"  # 16 ratings: sufficient for required player-library case.

    print(f"STEP 3: {len(items)} games, {ratings['user_id'].nunique()} players, "
          f"{len(ratings)} training ratings")
    print(f"\n1. Popularity baseline: Top-5 unplayed games for {user_id}")
    print_recommendations(recommender.recommend_popularity(user_id=user_id))
    print(f"\n2. NLP query: {query!r}")
    print_recommendations(recommender.recommend_content(query=query))
    print(f"\n3. Content-based user profile: Top-5 for {user_id}")
    print_recommendations(recommender.recommend_content(user_id=user_id))
    print(f"\n4. Hybrid (user CF + content + review sentiment): Top-5 for {user_id}")
    print_recommendations(recommender.recommend_hybrid(user_id=user_id))
    print("\n5. Cold-start example: upcoming indie game with no ratings")
    recommender.add_cold_start_game(
        item_id="NEW001", title="Forestcraft Horizons (fictional indie game)",
        genre_tags="open world|exploration|crafting|survival|story rich",
        store_description=(
            "Explore a story driven open world forest, craft tools, build camps, "
            "and uncover mysteries through survival adventures."
        ),
    )
    matches = recommender.recommend_content(query=query)
    print_recommendations(matches)
    print(f"  New indie game discovered in Top-5: "
          f"{any(hit['item_id'] == 'NEW001' for hit in matches)}")


if __name__ == "__main__":
    main()
