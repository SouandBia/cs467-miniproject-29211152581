"""Step 4: leakage-safe offline evaluation and required demonstration cases.

Run from the project root:
    python -m src.evaluate

Protocol:
  * Within each user, hide ~20% of interactions (3 out of 16 per user).
  * Fit every recommender on TRAINING RATINGS ONLY. Held-out ratings/reviews
    must never enter user history, popularity scores, CF, or sentiment.
  * Evaluate Top-5 unplayed rankings for relevant hidden games (rating >= 4).
  * Mean Precision@5 includes all eligible test users, even those without
    held-out positives. Recall@5 is computed only for users with positives.
  * Report collaborative-filtering rating MAE/RMSE separately where coverage
    exists; hybrid ranking scores are NOT predictions on the 1-5 scale.

The three course demonstration cases are run on the *complete supplied data*
AFTER offline evaluation, and are not reported as independent test accuracy.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Save charts even without a desktop display.
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .recommender import GameRecommender

PROJECT_ROOT = Path(__file__).resolve().parents[1]
GAME_QUERY = "story driven open world exploration with crafting"
METHODS = ("Popularity", "Content-Based", "Hybrid")
COLD_ITEM_ID = "NEW001"


def split_ratings_by_user(
    ratings: pd.DataFrame, *, test_fraction: float = 0.20,
    min_train_per_user: int = 6, seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Deterministically hold out interactions separately for each player.

    Unlike a global random split, this keeps at least min_train_per_user
    observations for every user with enough ratings. No duplicate user/item
    pairs should exist. Short-history users stay in training, untested.
    """
    if not 0 < test_fraction < 1:
        raise ValueError("test_fraction must be between zero and one")
    if min_train_per_user < 1:
        raise ValueError("min_train_per_user must be >= 1")
    required = {"user_id", "item_id", "rating", "player_review"}
    if not required.issubset(ratings.columns):
        raise ValueError(f"Missing columns: {sorted(required - set(ratings.columns))}")
    if ratings.duplicated(["user_id", "item_id"]).any():
        raise ValueError("Duplicate user-item rows are not supported")
    rng = np.random.default_rng(seed)
    test_indices: list[int] = []
    # Sorting makes the split independent of the CSV row ordering.
    for _, group in ratings.groupby("user_id", sort=True):
        ordered = group.sort_values("item_id", kind="stable")
        count = len(ordered)
        test_count = min(max(1, round(count * test_fraction)), count - min_train_per_user)
        if test_count <= 0:
            continue
        chosen = rng.choice(ordered.index.to_numpy(), size=test_count, replace=False)
        test_indices.extend(chosen.tolist())
    test = ratings.loc[sorted(test_indices)].reset_index(drop=True).copy()
    train = ratings.drop(index=test_indices).reset_index(drop=True).copy()
    return train, test


def precision_at_k(predicted_ids: list[str], relevant_ids: set[str], k: int = 5) -> float:
    """Precision@k = relevant recommendations / k, including empty lists."""
    if k <= 0:
        raise ValueError("k must be positive")
    return len(set(predicted_ids[:k]) & relevant_ids) / k


def recall_at_k(predicted_ids: list[str], relevant_ids: set[str], k: int = 5) -> float:
    """Recall@k = relevant recommendations / all held-out relevant games."""
    if k <= 0:
        raise ValueError("k must be positive")
    return (len(set(predicted_ids[:k]) & relevant_ids) / len(relevant_ids)
            if relevant_ids else float("nan"))


def evaluate_recommenders(
    items: pd.DataFrame, train: pd.DataFrame, test: pd.DataFrame,
    *, k: int = 5, positive_threshold: int = 4,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Evaluate all three methods on the *same* users and held-out items.

    Returns (method summary, per-user rows, CF rating-prediction diagnostics).
    The diagnostic CF predictions are from training data only; users/games
    without a CF prediction are counted as uncovered, not silently imputed.
    """
    if k < 1:
        raise ValueError("k must be positive")
    overlap = pd.merge(train[["user_id", "item_id"]],
                       test[["user_id", "item_id"]],
                       on=["user_id", "item_id"])
    if not overlap.empty:
        raise ValueError("Train/test leakage: at least one pair appears in both")
    engine = GameRecommender().fit(items, train)
    per_user: list[dict[str, object]] = []
    cf_pairs: list[dict[str, object]] = []
    train_users = set(train["user_id"].astype(str))

    for user_id, rows in test.groupby("user_id", sort=True):
        user_id = str(user_id)
        if user_id not in train_users:
            continue
        seen = set(train.loc[train["user_id"].astype(str) == user_id, "item_id"].astype(str))
        relevant = set(rows.loc[rows["rating"] >= positive_threshold, "item_id"].astype(str))
        rankings = {
            "Popularity": engine.recommend_popularity(user_id=user_id, top_k=k),
            "Content-Based": engine.recommend_content(user_id=user_id, top_k=k),
            "Hybrid": engine.recommend_hybrid(user_id=user_id, top_k=k),
        }
        for method, hits in rankings.items():
            predicted = [str(hit["item_id"]) for hit in hits]
            if seen.intersection(predicted):
                raise AssertionError(f"{method} recommended a training game to {user_id}")
            hits_count = len(set(predicted) & relevant)
            per_user.append({
                "user_id": user_id, "method": method,
                "n_hidden": len(rows), "n_relevant_hidden": len(relevant),
                "recommendations": len(predicted), "relevant_hits": hits_count,
                "precision_at_5": precision_at_k(predicted, relevant, k),
                "recall_at_5": recall_at_k(predicted, relevant, k),
                "hit_at_5": float(hits_count > 0),
            })
        predictions = engine.collaborative_predictions(user_id)
        for _, row in rows.iterrows():
            game = str(row["item_id"])
            if game in predictions:
                predicted_rating = float(predictions[game])
                cf_pairs.append({
                    "user_id": user_id, "item_id": game,
                    "actual_rating": float(row["rating"]),
                    "predicted_rating": predicted_rating,
                    "absolute_error": abs(float(row["rating"]) - predicted_rating),
                    "squared_error": (float(row["rating"]) - predicted_rating) ** 2,
                })

    details = pd.DataFrame(per_user)
    if details.empty:
        raise ValueError("No evaluable test users. Supply users with more ratings.")
    summary = []
    for method in METHODS:
        rows = details.loc[details.method == method]
        summary.append({
            "method": method,
            "users_evaluated": len(rows),
            "users_with_relevant_test_items": int((rows.n_relevant_hidden > 0).sum()),
            "precision_at_5": float(rows.precision_at_5.mean()),
            "recall_at_5": (float(rows.recall_at_5.mean())
                            if rows.recall_at_5.notna().any() else float("nan")),
            "hit_rate_at_5": float(rows.hit_at_5.mean()),
            "mean_returned_recommendations": float(rows.recommendations.mean()),
        })
    cf_frame = pd.DataFrame(cf_pairs, columns=[
        "user_id", "item_id", "actual_rating", "predicted_rating",
        "absolute_error", "squared_error",
    ])
    return pd.DataFrame(summary), details, cf_frame


def make_charts(ratings: pd.DataFrame, summary: pd.DataFrame, results_dir: Path) -> list[Path]:
    """Save assignment-required rating distribution and method comparison PNGs."""
    results_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 10, "figure.dpi": 130})

    counts = ratings["rating"].value_counts().reindex([1, 2, 3, 4, 5], fill_value=0)
    fig, ax = plt.subplots(figsize=(8, 4.8), constrained_layout=True)
    bars = ax.bar([str(r) for r in counts.index], counts.values)
    ax.bar_label(bars, padding=3)
    ax.set_title("Player rating distribution (synthetic dataset)", fontweight="bold", pad=14)
    ax.set_xlabel("Rating (1–5 stars)")
    ax.set_ylabel("Number of ratings")
    ax.set_ylim(0, max(counts) * 1.20)
    ax.grid(axis="y", alpha=0.20)
    ax.set_axisbelow(True)
    rating_path = results_dir / "rating_distribution.png"
    fig.savefig(rating_path, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.8, 4.8), constrained_layout=True)
    values = summary["precision_at_5"].to_numpy(dtype=float)
    bars = ax.bar(summary["method"], values)
    ax.bar_label(bars, labels=[f"{v:.3f}" for v in values], padding=4)
    ax.set_title("Offline Top-5 precision by recommendation method", fontweight="bold", pad=14)
    ax.set_ylabel("Mean Precision@5 (relevant = 4–5 stars)")
    ax.set_ylim(0, max(0.12, float(np.max(values)) * 1.28))
    ax.grid(axis="y", alpha=0.20)
    ax.set_axisbelow(True)
    ax.set_xlabel("Same per-user 20% holdout for all methods; simulated ratings", labelpad=12)
    precision_path = results_dir / "precision_at_5_comparison.png"
    fig.savefig(precision_path, bbox_inches="tight")
    plt.close(fig)
    return [rating_path, precision_path]


def format_matches(matches: list[dict]) -> list[str]:
    """Format ranked games with their keywords for saved demo evidence."""
    if not matches:
        return ["  (No recommendations)"]
    output = []
    for rank, hit in enumerate(matches, start=1):
        keywords = ", ".join(hit.get("keywords", [])) or "none (CF/review-driven match)"
        evidence = []
        if hit.get("predicted_rating") is not None:
            evidence.append(f"CF estimated rating={hit['predicted_rating']:.2f}/5")
        if hit.get("review_sentiment") is not None:
            evidence.append(f"review sentiment={hit['review_sentiment']:+.2f}")
        evidence_text = ("; " + "; ".join(evidence)) if evidence else ""
        output.append(f"  {rank}. {hit['title']} [{hit['item_id']}] "
                      f"score={hit['score']:.4f}; keywords={keywords}{evidence_text}")
    return output


def run_three_test_cases(items: pd.DataFrame, ratings: pd.DataFrame, results_dir: Path,
                         *, player_id: str = "U01") -> dict[str, object]:
    """Three assignment demonstrations using full data, NOT held-out evaluation.

    Cold-start adds one fictional game with tags/description only and verifies
    that it appears among top five query results without any player ratings.
    """
    results_dir.mkdir(parents=True, exist_ok=True)
    player_ratings = int((ratings["user_id"].astype(str) == player_id).sum())
    if player_ratings < 6:
        raise ValueError("Player library test requires at least 6 observed games")
    model = GameRecommender().fit(items, ratings)
    query_hits = model.recommend_content(query=GAME_QUERY, top_k=5)
    player_hits = model.recommend_hybrid(user_id=player_id, top_k=5)
    rated_ids = set(ratings.loc[ratings.user_id.astype(str) == player_id, "item_id"].astype(str))
    assert len(query_hits) == 5 and len(player_hits) == 5
    assert rated_ids.isdisjoint(hit["item_id"] for hit in player_hits)

    new_game_name = "Forestcraft Horizons (fictional indie game)"
    model.add_cold_start_game(
        item_id=COLD_ITEM_ID, title=new_game_name,
        genre_tags="open world|exploration|crafting|survival|story rich",
        store_description=(
            "Explore a story driven open world forest, craft tools, build camps, "
            "and uncover mysteries through survival adventures."
        ),
        platform="PC", play_mode="single-player",
    )
    cold_hits = model.recommend_content(query=GAME_QUERY, top_k=5)
    cold_ids = [hit["item_id"] for hit in cold_hits]
    appeared = COLD_ITEM_ID in cold_ids
    no_ratings = COLD_ITEM_ID not in set(model.ratings["item_id"].astype(str))
    assert appeared and no_ratings, "Cold-start game must be discoverable without ratings"

    lines = [
        "CS467 MP-A12 — STEP 4 REQUIRED TEST CASES",
        "NOTE: Examples use all available data. They are demonstrations, NOT offline test metrics.",
        "",
        f"CASE 1 — Gameplay query: {GAME_QUERY!r}",
        *format_matches(query_hits), "",
        f"CASE 2 — Player library: Top-5 unrated games for {player_id} "
        f"({player_ratings} previously rated games), via hybrid recommender",
        *format_matches(player_hits),
        f"  All five games excluded from {player_id}'s existing library: "
        f"{rated_ids.isdisjoint(hit['item_id'] for hit in player_hits)}", "",
        f"CASE 3 — Cold-start: Added {new_game_name} [{COLD_ITEM_ID}] "
        "with metadata only and no ratings",
        *format_matches(cold_hits),
        f"  Cold-start game appears in top five: {appeared}",
        f"  Cold-start game has zero player ratings: {no_ratings}",
        f"  Cold-start game rank: {cold_ids.index(COLD_ITEM_ID) + 1}", "",
    ]
    (results_dir / "sample_outputs.txt").write_text("\n".join(lines), encoding="utf-8")
    return {
        "query_top5": query_hits,
        "player_top5": player_hits,
        "cold_start_top5": cold_hits,
        "cold_start_found": appeared,
        "cold_start_zero_ratings": no_ratings,
        "player_rated_count": player_ratings,
    }


def run_evaluation(root: Path = PROJECT_ROOT, *, seed: int = 42) -> dict[str, object]:
    """Run all Step 4 artifacts and return metrics for tests / CLI scripts."""
    items = pd.read_csv(root / "data" / "items.csv")
    ratings = pd.read_csv(root / "data" / "ratings.csv")
    results = root / "results"
    results.mkdir(parents=True, exist_ok=True)
    train, test = split_ratings_by_user(ratings, seed=seed)
    summary, per_user, cf_predictions = evaluate_recommenders(items, train, test)
    chart_paths = make_charts(ratings, summary, results)
    cases = run_three_test_cases(items, ratings, results)

    summary.round(5).to_csv(results / "evaluation_metrics.csv", index=False)
    per_user.round(5).to_csv(results / "per_user_metrics.csv", index=False)
    # Store only the IDs of holdout pairs (not review contents) for auditability.
    test[["user_id", "item_id", "rating"]].to_csv(
        results / "holdout_test_interactions.csv", index=False
    )
    cf_predictions.round(5).to_csv(results / "cf_rating_predictions.csv", index=False)
    if not cf_predictions.empty:
        mae = float(cf_predictions.absolute_error.mean())
        rmse = float(np.sqrt(cf_predictions.squared_error.mean()))
    else:
        mae = rmse = float("nan")
    report = [
        "CS467 MP-A12 — STEP 4 OFFLINE EVALUATION",
        f"Seed: {seed}; per-player ~20% holdout; train ratings: {len(train)}; "
        f"test ratings: {len(test)}",
        "All three algorithms are fitted using ONLY training ratings and reviews.",
        "Relevant game = hidden game with rating >= 4/5; all predictions exclude training games.",
        "Precision@5 = number of relevant hidden recommendations / 5, macro-averaged over test users.",
        "Recall@5 = relevant hidden recommendations / number of hidden relevant games; "
        "only users with hidden relevant games are included.",
        "Hit Rate@5 = fraction of test users with >= 1 hidden relevant game recommended.",
        "", summary.to_string(index=False, float_format=lambda val: f"{val:.4f}"), "",
        f"User-based CF numeric rating estimates: coverage="
        f"{len(cf_predictions)}/{len(test)} held-out ratings "
        f"({len(cf_predictions)/len(test):.1%}), "
        f"MAE={mae:.4f}, RMSE={rmse:.4f} (measured on covered pairs only).",
        "Hybrid/content ranking scores are NOT numeric rating predictions and have no RMSE/MAE.",
        "", "CAUTIONS:",
        "- All ratings and player reviews are synthetic; rankings do not prove real-world accuracy.",
        "- With only three hidden interactions per user, Precision@5 is noisy and small.",
        "- Top-5 sample queries / cold-start examples were run separately on the full dataset,",
        "  not used to calculate the held-out ranking metrics.",
        "- The holdout contains only observed ratings; unknown preferences are unmeasured.",
        "- Small training overlaps limit user-based CF; missing predictions affect coverage.", "",
    ]
    (results / "evaluation_report.txt").write_text("\n".join(report), encoding="utf-8")
    return {
        "summary": summary, "per_user": per_user, "train": train, "test": test,
        "cf_predictions": cf_predictions, "mae": mae, "rmse": rmse,
        "charts": chart_paths, "cases": cases, "report": "\n".join(report),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the CS467 game recommenders")
    parser.add_argument("--seed", type=int, default=42, help="Random holdout seed (default: 42)")
    args = parser.parse_args()
    report = run_evaluation(seed=args.seed)
    print(report["report"])
    print("Results saved in:", PROJECT_ROOT / "results")
    print("Required demonstration cases:", PROJECT_ROOT / "results" / "sample_outputs.txt")
    print("Charts:", ", ".join(path.name for path in report["charts"]))


if __name__ == "__main__":
    main()
