"""Build the rebtopk token lists from the saved census into `data/gkd_vocab_mask_lists/`.

    uv run python -m scripts.build_rebtopk_token_lists

Writes one JSON per list and prints each list's ids sha256 for `rebtopk_plan`.
"""

from __future__ import annotations

from pipelines.rebtopk_lists_pipeline import build_lists


def main() -> None:
    build_lists()


if __name__ == "__main__":
    main()
