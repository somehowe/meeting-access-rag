"""Compare retrieval methods and report abstention on labeled questions."""

from __future__ import annotations

import argparse
import json
from statistics import mean

from meeting_rag import DATA, DEFAULT_MIN_SCORE, DEFAULT_TOP_K, Index, load_passages


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--show-misses", action="store_true")
    args = parser.parse_args()
    if args.k < 1:
        parser.error("--k must be positive")

    passages = load_passages(DATA / "docs.json")
    index = Index(passages)
    cases = json.loads((DATA / "eval.json").read_text(encoding="utf-8"))
    known_ids = {passage.id for passage in passages}
    for case in cases:
        unknown_ids = set(case["relevant"]) - known_ids
        if unknown_ids:
            parser.error(f"Unknown passage IDs in evaluation: {sorted(unknown_ids)}")
    answerable = [case for case in cases if case["relevant"]]
    unanswerable = [case for case in cases if not case["relevant"]]
    print(f"{len(answerable)} answerable queries; {len(unanswerable)} unanswerable queries")
    print(f"{'method':<16} {'recall@' + str(args.k):>10} {'MRR@' + str(args.k):>10}")
    for method in ("overlap", "bm25"):
        recalls, reciprocal_ranks = [], []
        for case in answerable:
            hits = index.search(case["question"], args.k, method=method)
            ranked_ids = [hit.passage.id for hit in hits]
            relevant = set(case["relevant"])
            found = relevant.intersection(ranked_ids)
            recalls.append(len(found) / len(relevant))
            reciprocal_ranks.append(
                next((1 / rank for rank, doc in enumerate(ranked_ids, 1) if doc in relevant), 0)
            )
            if args.show_misses and relevant - found:
                print(f"  {method} missed {sorted(relevant - found)}: {case['question']}")
        print(f"{method:<16} {mean(recalls):>10.3f} {mean(reciprocal_ranks):>10.3f}")

    answered = sum(
        bool(index.search(case["question"], args.k, min_score=DEFAULT_MIN_SCORE))
        for case in answerable
    )
    abstained = sum(
        not index.search(case["question"], args.k, min_score=DEFAULT_MIN_SCORE)
        for case in unanswerable
    )
    print(f"BM25 with min score {DEFAULT_MIN_SCORE:g}: evidence returned for "
          f"{answered}/{len(answerable)} answerable queries; "
          f"abstained on {abstained}/{len(unanswerable)} unanswerable queries.")


if __name__ == "__main__":
    main()
