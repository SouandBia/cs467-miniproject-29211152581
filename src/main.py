"""CLI entry point for the CS467 MP-A12 Video Game Description and Tag Recommender.

Run from the project root:
    python -m src.main
    python -m src.main --user U01 --method hybrid
    python -m src.main --query "story driven open world exploration with crafting"
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .recommender import GameRecommender, print_recommendations

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_QUERY = "story driven open world exploration with crafting"


def _run_default_demo(recommender: GameRecommender, *, user_id: str, query: str, top_k: int) -> None:
    """Run all three assignment test cases plus popularity and content baselines."""
    print(f"=== 1. Gameplay Query (Content-Based): {query!r} ===")
    print_recommendations(recommender.recommend_content(query=query, top_k=top_k))

    print(f"\n=== 2a. Popularity Baseline for {user_id} ===")
    print_recommendations(recommender.recommend_popularity(user_id=user_id, top_k=top_k))

    print(f"\n=== 2b. Content-Based Profile for {user_id} ===")
    print_recommendations(recommender.recommend_content(user_id=user_id, top_k=top_k))

    print(f"\n=== 2c. Player Library Test (Hybrid) for {user_id} ===")
    print_recommendations(recommender.recommend_hybrid(user_id=user_id, top_k=top_k))

    print("\n=== 3. Cold-Start Test (New Unrated Indie Game) ===")
    recommender.add_cold_start_game(
        item_id="NEW001",
        title="Forestcraft Horizons (fictional indie game)",
        genre_tags="open world|exploration|crafting|survival|story rich",
        store_description=(
            "Explore a story driven open world forest, craft tools, build camps, "
            "and uncover mysteries through survival adventures."
        ),
        platform="PC",
        play_mode="single-player",
    )
    cold_hits = recommender.recommend_content(query=query, top_k=top_k)
    print_recommendations(cold_hits)
    found = any(hit["item_id"] == "NEW001" for hit in cold_hits)
    print(f"  Cold-start game discovered in Top-{top_k}: {found}")


def main() -> None:
    """Parse CLI arguments and run game recommendations."""
    parser = argparse.ArgumentParser(
        description="CS467 MP-A12 Video Game Description and Tag Recommender"
    )
    parser.add_argument("--user", type=str, default=None, help="Target player ID (e.g., U01)")
    parser.add_argument("--query", type=str, default=None, help="Free-text gameplay query")
    parser.add_argument(
        "--method",
        choices=("popularity", "content", "hybrid"),
        default="hybrid",
        help="Recommendation method when --user is provided (default: hybrid)",
    )
    parser.add_argument("--top-k", type=int, default=5, help="Number of recommendations (default: 5)")
    args = parser.parse_args()

    items = pd.read_csv(PROJECT_ROOT / "data" / "items.csv")
    ratings = pd.read_csv(PROJECT_ROOT / "data" / "ratings.csv")
    recommender = GameRecommender().fit(items, ratings)

    if args.query is not None:
        print(f"Top-{args.top_k} Content-Based Recommendations for query: {args.query!r}")
        print_recommendations(recommender.recommend_content(query=args.query, top_k=args.top_k))
        return

    if args.user is not None:
        if args.method == "popularity":
            hits = recommender.recommend_popularity(user_id=args.user, top_k=args.top_k)
        elif args.method == "content":
            hits = recommender.recommend_content(user_id=args.user, top_k=args.top_k)
        else:
            hits = recommender.recommend_hybrid(user_id=args.user, top_k=args.top_k)
        print(f"Top-{args.top_k} {args.method.title()} Recommendations for {args.user}:")
        print_recommendations(hits)
        return

    _run_default_demo(recommender, user_id="U01", query=DEFAULT_QUERY, top_k=args.top_k)


if __name__ == "__main__":
    main()
