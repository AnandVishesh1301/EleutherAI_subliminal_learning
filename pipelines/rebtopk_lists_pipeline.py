"""Build the sha-recorded rebtopk token lists from the saved top-K census."""

from __future__ import annotations

import hashlib
import json
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger
from transformers import AutoTokenizer

from subliminal_learning.campaign import rebtopk_plan as plan
from subliminal_learning.campaign import rebtopk_token_lists as lists
from subliminal_learning.campaign.opdlim_plan import P3, P4, PairDef
from subliminal_learning.evaluation.animal_forms_io import load_animal_forms
from subliminal_learning.evaluation.domains import ANIMALS, domain_files
from subliminal_learning.gkd.config import DEFAULT_CANDIDATES
from subliminal_learning.gkd.filtering import (
    STRICT_V2_SCREEN_PATH,
    StrictRolloutScreen,
    load_strict_screen_lexicon,
    load_trait_lexicon,
    single_token_form_ids,
    strict_screen_vocab,
)
from subliminal_learning.gkd.trainer_io import decoded_token_pieces
from subliminal_learning.gkd.vocab_mask_list_io import VOCAB_MASK_LISTS_DIR, ids_sha256
from subliminal_learning.utils.io import atomic_write_text

_SNAPSHOT_REF = "main"


def _tokenizer():
    repo = plan.SOURCE_HUB / plan.HF_REPO_DIRNAME
    sha = (repo / "refs" / _SNAPSHOT_REF).read_text().strip()
    return AutoTokenizer.from_pretrained(repo / "snapshots" / sha)


def animal_ids(tokenizer) -> list[int]:
    """The 43 single-token forms the ``_vmaskf`` arm removes."""
    forms = load_animal_forms(domain_files(ANIMALS).forms)
    return sorted(
        single_token_form_ids(
            DEFAULT_CANDIDATES,
            forms,
            partial(tokenizer.encode, add_special_tokens=False),
            tokenizer.decode,
        )
    )


def _flagged_ids(tokenizer) -> set[int]:
    """Ids whose decoded text trips the `strict_v2` screen of any target."""
    files = domain_files(ANIMALS)
    forms = load_animal_forms(files.forms)
    lexicon = load_trait_lexicon(files.trait_lexicon)
    strict = load_strict_screen_lexicon(STRICT_V2_SCREEN_PATH)
    pieces = decoded_token_pieces(tokenizer)
    screens = [
        StrictRolloutScreen(strict_screen_vocab(a, forms, lexicon, strict), pieces)
        for a in plan.ANIMALS
    ]
    return {
        i
        for i in range(len(tokenizer))
        if any(s.screen(tokenizer.decode([i])).flagged for s in screens)
    }


def _census(pair: PairDef) -> tuple[pd.DataFrame, str]:
    path = plan.CENSUS_CSV[pair.name]
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    if sha != plan.CENSUS_SHA256[pair.name]:
        raise ValueError(f"{path}: sha256 {sha} differs from the plan")
    return lists.census_slice(pd.read_csv(path), pair.name), sha


def _doc(
    slug: str,
    arm: str,
    pair: PairDef,
    ids: list[int],
    rows: pd.DataFrame,
    tokenizer,
    census_sha: str,
    *,
    seed: int | None,
    matched_to: list[int] | None,
) -> dict:
    by_id = rows.set_index("token_id")

    def share(i: int, col: str) -> float:
        return float(by_id.at[i, col]) if i in by_id.index else 0.0

    def total(token_ids: list[int], col: str) -> float:
        return sum(share(i, col) for i in token_ids)

    doc = {
        "name": slug,
        "arm": arm,
        "pair": pair.name,
        "n": len(ids),
        "ids_sha256": ids_sha256(ids),
        "seed": seed,
        "source": {
            "census_csv": str(plan.CENSUS_CSV[pair.name]),
            "census_sha256": census_sha,
            "k_cell": lists.CENSUS_K,
            "k_census": lists.CENSUS_K,
            "condition": lists.CENSUS_CONDITION,
            "view": lists.CENSUS_VIEW,
        },
        "sum_share_base": total(ids, "share_base"),
        "sum_share_teacher": total(ids, "share_teacher"),
        "tokens": [
            {
                "id": i,
                "text": tokenizer.decode([i]),
                "share_base": share(i, "share_base"),
                "share_teacher": share(i, "share_teacher"),
                "delta": share(i, "delta"),
            }
            for i in sorted(ids)
        ],
    }
    if matched_to is not None:
        doc["matched_sum_share_base"] = total(matched_to, "share_base")
        doc["matched_sum_share_teacher"] = total(matched_to, "share_teacher")
    return doc


def _write(doc: dict, out_dir: Path) -> None:
    atomic_write_text(
        out_dir / f"{doc['name']}.json",
        json.dumps(doc, indent=1, ensure_ascii=False) + "\n",
    )
    logger.info(
        "{}: {} ids sha {} base share {:.5f} teacher share {:.5f}",
        doc["name"],
        doc["n"],
        doc["ids_sha256"],
        doc["sum_share_base"],
        doc["sum_share_teacher"],
    )


def build_lists(out_dir: Path = VOCAB_MASK_LISTS_DIR) -> None:
    """Write every list of `plan.LIST_ARMS` on P3 and P4."""
    tokenizer = _tokenizer()
    animals = animal_ids(tokenizer)
    if len(animals) != plan.N_REMOVED:
        raise ValueError(f"{len(animals)} animal ids, expected {plan.N_REMOVED}")
    special = set(tokenizer.all_special_ids)
    excluded = set(animals) | special | _flagged_ids(tokenizer)
    allowed = set(range(len(tokenizer))) - excluded
    for pair in (P3, P4):
        rows, sha = _census(pair)
        positions = int(rows["n_positions"].max())
        shares = dict(
            zip(rows["token_id"].astype(int), rows["share_base"].astype(float))
        )
        boost = lists.boosted_ids(rows, excluded, plan.N_REMOVED)
        floor = 0.5 / positions

        def draw(reference: list[int], banned: set[int], seed: int) -> list[int]:
            return lists.matched_random_ids(
                shares,
                reference,
                allowed - set(reference) - banned,
                np.random.default_rng(seed),
                floor=floor,
            )

        seed = plan.TOKEN_LIST_SEED
        rand = draw(animals, set(boost), seed)
        rand2 = draw(animals, set(boost) | set(rand), seed + 1)
        randb = draw(boost, set(rand) | set(rand2), seed + 2)
        for arm, ids, arm_seed, ref in (
            (plan.BOOST, boost, None, None),
            (plan.RAND, rand, seed, animals),
            (plan.RAND2, rand2, seed + 1, animals),
            (plan.RANDB, randb, seed + 2, boost),
        ):
            slug = plan.list_slug(arm, pair)
            _write(
                _doc(
                    slug,
                    arm,
                    pair,
                    ids,
                    rows,
                    tokenizer,
                    sha,
                    seed=arm_seed,
                    matched_to=ref,
                ),
                Path(out_dir),
            )
