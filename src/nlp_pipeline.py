"""NLP features for the CS467 video-game recommendation mini-project.

This module implements the assignment's NLP requirements:
  * lowercase, punctuation/whitespace cleanup, tokenization, stop-word removal
  * unigram + bigram TF-IDF from game descriptions and genre tags
  * transforming unseen text with the *same fitted* TF-IDF vocabulary
  * keyword explanations for cosine-similarity matches
  * basic positive/negative sentiment ratios for player reviews

This is the content-processing layer. Popularity, collaborative filtering,
hybrid scoring, and holdout evaluation belong to later project stages.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

REQUIRED_ITEM_COLUMNS = (
    "item_id", "title", "platform", "genre_tags", "store_description", "play_mode"
)
REQUIRED_RATING_COLUMNS = ("user_id", "item_id", "rating", "player_review")

# Small, explicit lexicons appropriate to the simulated short reviews in the dataset.
# For more complex real-world reviews, a trained sentiment classifier would be better.
POSITIVE_WORDS = frozenset({
    "amazing", "awesome", "beautiful", "brilliant", "decent", "easy", "engaging",
    "enjoy", "enjoyable", "enjoyed", "excellent", "exciting", "fantastic", "favorite",
    "fun", "good", "great", "impressive", "interesting", "love", "loved", "memorable",
    "outstanding", "polished", "recommend", "rewarding", "satisfying", "smooth",
    "superb", "wonderful",
})
NEGATIVE_WORDS = frozenset({
    "annoying", "awful", "bad", "boring", "broken", "bug", "bugs", "confusing",
    "crash", "crashes", "disappointing", "dull", "frustrating", "hate", "hated",
    "laggy", "poor", "repetitive", "slow", "tedious", "terrible", "unbalanced",
    "uneven", "unfair", "unstable", "weak", "worse", "worst",
})
NEGATION_WORDS = frozenset({"not", "no", "never", "hardly"})


def clean_text(text: Any) -> str:
    """Lowercase, remove punctuation, and collapse extra whitespace.

    Example: ``'Open-World, Crafting!  '`` -> ``'open world crafting'``.
    """
    if text is None or (not isinstance(text, str) and pd.isna(text)):
        return ""
    lowered = str(text).lower().replace("_", " ")
    letters_numbers_spaces = re.sub(r"[^\w\s]", " ", lowered, flags=re.UNICODE)
    return re.sub(r"\s+", " ", letters_numbers_spaces).strip()


def tokenize(text: Any, *, remove_stopwords: bool = True) -> list[str]:
    """Split cleaned text into tokens and optionally discard English stop words."""
    tokens = clean_text(text).split()
    if remove_stopwords:
        return [token for token in tokens if token not in ENGLISH_STOP_WORDS]
    return tokens


def review_sentiment(review: Any) -> dict[str, float | int | str]:
    """Calculate positive/negative evidence ratios using a simple word lexicon.

    - positive_ratio = positive_count / (positive_count + negative_count)
    - negative_ratio = negative_count / (positive_count + negative_count)
    - sentiment_score = positive_ratio - negative_ratio, in [-1, +1]
    - label: positive, negative, mixed, or neutral

    No sentiment words => ratios and score are 0, label 'neutral'.
    A single word immediately following a negation is reversed, e.g. 'not good'.
    """
    words = tokenize(review, remove_stopwords=False)
    positive = negative = 0
    for index, word in enumerate(words):
        # Flip sentiment if the immediately preceding word negates it.
        negated = index > 0 and words[index - 1] in NEGATION_WORDS
        if word in POSITIVE_WORDS:
            negative += int(negated)
            positive += int(not negated)
        elif word in NEGATIVE_WORDS:
            positive += int(negated)
            negative += int(not negated)

    total = positive + negative
    pos_ratio = positive / total if total else 0.0
    neg_ratio = negative / total if total else 0.0
    score = pos_ratio - neg_ratio
    label = ("neutral" if total == 0 else
             "positive" if positive > negative else
             "negative" if negative > positive else "mixed")
    return {
        "positive_count": positive,
        "negative_count": negative,
        "positive_ratio": round(pos_ratio, 4),
        "negative_ratio": round(neg_ratio, 4),
        "sentiment_score": round(score, 4),
        "sentiment_label": label,
    }


class GameNLPPipeline:
    """Fit one shared vocabulary to games and transform queries consistently."""

    def __init__(self) -> None:
        self.vectorizer = TfidfVectorizer(
            tokenizer=tokenize,
            token_pattern=None,  # Custom tokenizer handles token boundaries.
            lowercase=False,  # Already handled by clean_text().
            ngram_range=(1, 2),  # Unigrams and bigrams.
            norm="l2",  # Makes cosine similarity straightforward.
            sublinear_tf=True,
        )
        self.items: pd.DataFrame | None = None
        self.tfidf_matrix: csr_matrix | None = None
        self.item_index: dict[str, int] = {}
        self.review_features: pd.DataFrame | None = None

    def fit(
        self, items: pd.DataFrame, ratings: pd.DataFrame | None = None
    ) -> "GameNLPPipeline":
        """Learn TF-IDF on item descriptions and tags, optionally score reviews."""
        missing = set(REQUIRED_ITEM_COLUMNS).difference(items.columns)
        if missing:
            raise ValueError(f"Missing columns in items.csv: {sorted(missing)}")
        if items.empty:
            raise ValueError("items.csv cannot be empty")
        if items["item_id"].isna().any() or items["item_id"].duplicated().any():
            raise ValueError("Each item_id must be unique and nonempty")

        self.items = items.reset_index(drop=True).copy()
        self.item_index = {
            str(item_id): index
            for index, item_id in enumerate(self.items["item_id"])
        }
        descriptions = (
            self.items["genre_tags"].fillna("").astype(str).str.replace("|", " ", regex=False)
            + " " + self.items["store_description"].fillna("").astype(str)
        )
        self.tfidf_matrix = self.vectorizer.fit_transform(descriptions).tocsr()

        self.review_features = None
        if ratings is not None:
            missing = set(REQUIRED_RATING_COLUMNS).difference(ratings.columns)
            if missing:
                raise ValueError(f"Missing columns in ratings.csv: {sorted(missing)}")
            self.review_features = ratings.reset_index(drop=True).copy()
            sentiment_columns = self.review_features["player_review"].apply(
                review_sentiment
            ).apply(pd.Series)
            self.review_features = pd.concat(
                [self.review_features, sentiment_columns], axis=1
            )
        return self

    def _check_fitted(self) -> None:
        if self.items is None or self.tfidf_matrix is None:
            raise RuntimeError("Call pipeline.fit(items, ratings) before requesting results")

    def transform_query(self, query: str) -> csr_matrix:
        """Vectorize new user text using the *fitted* vocabulary (no refitting)."""
        self._check_fitted()
        return self.vectorizer.transform([query]).tocsr()

    def matching_keywords(self, query_vector: csr_matrix, item_index: int,
                          top_n: int = 5) -> list[str]:
        """Return overlapping TF-IDF terms, ranked by contribution to similarity."""
        self._check_fitted()
        assert self.tfidf_matrix is not None
        overlap = query_vector.multiply(self.tfidf_matrix[item_index]).tocoo()
        if overlap.nnz == 0:
            return []
        features = self.vectorizer.get_feature_names_out()
        scored_terms = sorted(
            zip(overlap.col, overlap.data), key=lambda pair: -pair[1]
        )
        return [str(features[term_index]) for term_index, _ in scored_terms[:top_n]]

    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """Find Top-K games by cosine similarity to a free-text gameplay query."""
        self._check_fitted()
        assert self.items is not None and self.tfidf_matrix is not None
        if top_k < 1:
            return []
        query_vector = self.transform_query(query)
        if query_vector.nnz == 0:  # No words in vocabulary; avoid arbitrary matches.
            return []
        similarities = cosine_similarity(query_vector, self.tfidf_matrix).ravel()
        ranked = sorted(range(len(similarities)), key=lambda i: (-similarities[i], i))
        ranked = [i for i in ranked if similarities[i] > 0][:top_k]
        return [
            {
                "item_id": str(self.items.iloc[i]["item_id"]),
                "title": str(self.items.iloc[i]["title"]),
                "similarity": round(float(similarities[i]), 4),
                "keywords": self.matching_keywords(query_vector, i),
            }
            for i in ranked
        ]

    def similar_games(self, item_id: str, top_k: int = 5) -> list[dict[str, Any]]:
        """Find similar games using an existing game's TF-IDF description."""
        self._check_fitted()
        assert self.items is not None and self.tfidf_matrix is not None
        if item_id not in self.item_index:
            raise KeyError(f"Unknown game item_id: {item_id}")
        if top_k < 1:
            return []
        source_index = self.item_index[item_id]
        source_vector = self.tfidf_matrix[source_index]
        similarities = cosine_similarity(source_vector, self.tfidf_matrix).ravel()
        ranked = sorted(
            (i for i in range(len(similarities)) if i != source_index and similarities[i] > 0),
            key=lambda i: (-similarities[i], i),
        )[:top_k]
        return [
            {
                "item_id": str(self.items.iloc[i]["item_id"]),
                "title": str(self.items.iloc[i]["title"]),
                "similarity": round(float(similarities[i]), 4),
                "keywords": self.matching_keywords(source_vector, i),
            }
            for i in ranked
        ]

    def item_review_sentiment(self) -> pd.DataFrame:
        """Average review sentiment score and ratios for each rated game."""
        if self.review_features is None:
            raise RuntimeError("Pass ratings to fit() before aggregating sentiment")
        return (
            self.review_features.groupby("item_id", as_index=False)
            .agg(
                review_count=("player_review", "size"),
                mean_sentiment=("sentiment_score", "mean"),
                positive_ratio=("positive_ratio", "mean"),
                negative_ratio=("negative_ratio", "mean"),
            )
            .round(4)
        )


def print_top_matches(matches: list[dict[str, Any]]) -> None:
    """Display Top-K results with similarity and interpretable keywords."""
    if not matches:
        print("No matching games: try different gameplay keywords.")
    for rank, result in enumerate(matches, 1):
        keywords = ", ".join(result["keywords"]) or "no overlapping terms"
        print(f"{rank}. {result['title']} [{result['item_id']}] | "
              f"cosine similarity={result['similarity']:.4f} | matches: {keywords}")


def main() -> None:
    """Run a small demonstration of the NLP stage with provided CSV files."""
    data_dir = Path(__file__).resolve().parent.parent / "data"
    items = pd.read_csv(data_dir / "items.csv")
    ratings = pd.read_csv(data_dir / "ratings.csv")
    nlp = GameNLPPipeline().fit(items, ratings)

    query = "story driven open world exploration with crafting"
    print(f"Vocabulary size (unigrams + bigrams): {len(nlp.vectorizer.vocabulary_)}")
    print(f"\nGameplay query: {query!r}")
    print_top_matches(nlp.search(query))

    print("\nGames similar to Subnautica [G004]:")
    print_top_matches(nlp.similar_games("G004"))

    assert nlp.review_features is not None
    print("\nSentiment for sample player reviews:")
    for _, row in nlp.review_features.head(4).iterrows():
        print(f"  {row['player_review']} -> {row['sentiment_label']} "
              f"(pos={row['positive_ratio']:.2f}, neg={row['negative_ratio']:.2f})")


if __name__ == "__main__":
    main()
