"""Base-model roster of the ws160m campaign: the six Pythia-160M ablation categories.

Every base is described by three attributes fixed in
`docs/PYTHIA_160M_ABLATION_CELLS.md` (evidence in `docs/POLYPYTHIA_160M_SEED_AUDIT.md`):
its init, its batch stream and its corpus. A pair's category follows from which of the
first two the teacher's and the student's bases share; implementation (`H` Hub, `V`
`gpt-neox-v1.0`, `M` `gpt-neox-modern`) is a recorded nuisance attribute that never
defines a category. Hub and locally trained bases are both represented as
`HfModelSpec` entries, one per (base, step): the sweeps resolve `--base-models`
tokens through a single roster, all-or-nothing, so the two kinds cannot be mixed
otherwise.

Naming, never conflated: the **token** ``<base>@step<N>`` is the roster key carried by
sweep YAMLs and command lines; the **label** ``<base>-step<N>`` is the run-directory
segment (labels may not contain ``@`` or ``/``, and carry the step so rungs of one
base never share a directory); the **identity** ``model_ref`` is what manifests store
(`EleutherAI/pythia-160m@step143000`, `local/pile-ws1@step8000`).
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

from subliminal_learning.models.hf_spec import HfModelSpec

CATEGORIES: tuple[str, ...] = ("SI_SS", "SI_DS", "DI_SS", "DI_DS", "SI_DC", "DI_DC")
# Same init and same stream but different final weights (`pythia-160m` <-> `pile-main`):
# reported on its own, never pooled with the self-pairs (doc § 4.1).
SI_SS_NONIDENT = "SI_SS_NONIDENT"
IMPL_STRATA: tuple[str, ...] = (
    "hub_only",
    "local_hub",
    "local_local_same_checkout",
    "local_local_cross_stack",
)
HUB_ORG = "EleutherAI"
NEOX_TOKENIZER_SHA = "b0d09407f0017e4ddf710ae82c9ef07383b0b6d985425d5b07fcd931de7563d5"
NEOX_LORA: tuple[str, ...] = (
    "query_key_value",
    "dense",
    "dense_h_to_4h",
    "dense_4h_to_h",
)
VOCAB_SIZE = 50304
FINAL_STEP = 143000
BASE_MODELS_DIRNAME = "base_models_160m"
HF_CONVERTED_DIRNAME = "hf_converted"
HUB_CACHE_DIRNAME = "hf_cache"
WAVE_NOW, WAVE_2, WAVE_3 = 1, 2, 3
# Pretraining steps any ws160m / ws160m2 cell may load (ladder rungs plus the final step).
CAMPAIGN_STEPS: tuple[int, ...] = (
    1000,
    2000,
    4000,
    8000,
    16000,
    32000,
    48000,
    96000,
    143000,
)


@dataclass(frozen=True)
class BaseModel:
    """One 160M base model of the roster.

    Attributes:
        base: Inventory name (`docs/PYTHIA_160M_ABLATION_CELLS.md` § 3).
        source: ``"hub"`` or ``"local"``.
        hub_repo: Hub repo id for a Hub base, else ``None``.
        init: Init label (``I1234``, ``I1``..``I4``).
        stream: Batch-stream label (``P0``, ``PdN``, ``PjN``, ``C``).
        corpus: ``"pile"`` or ``"dclm"``.
        impl: ``"H"``, ``"V"`` or ``"M"``.
        checkout: Training code checkout of a local base (``"v1.0"``/``"modern"``), else
            ``None``; equal checkouts form the same-checkout stratum.
        wave: Campaign wave in which the base's step143000 becomes usable.
        ladder_cap: Highest staged pretraining step at staging time.
        init_twin: Hub base whose ``step0`` equals this local base's init (audit R3).
    """

    base: str
    source: str
    hub_repo: str | None
    init: str
    stream: str
    corpus: str
    impl: str
    checkout: str | None
    wave: int
    ladder_cap: int
    init_twin: str | None = None


def _hub(base: str, repo: str, init: str, stream: str) -> BaseModel:
    return BaseModel(
        base, "hub", repo, init, stream, "pile", "H", None, WAVE_NOW, FINAL_STEP
    )


ROSTER: tuple[BaseModel, ...] = (
    _hub("pythia-160m", f"{HUB_ORG}/pythia-160m", "I1234", "P0"),
    _hub("data-seed1", f"{HUB_ORG}/pythia-160m-data-seed1", "I1234", "Pd1"),
    _hub("data-seed2", f"{HUB_ORG}/pythia-160m-data-seed2", "I1234", "Pd2"),
    _hub("data-seed3", f"{HUB_ORG}/pythia-160m-data-seed3", "I1234", "Pd3"),
    _hub("seed1", f"{HUB_ORG}/pythia-160m-seed1", "I1", "Pj1"),
    _hub("seed2", f"{HUB_ORG}/pythia-160m-seed2", "I2", "Pj2"),
    _hub("seed3", f"{HUB_ORG}/pythia-160m-seed3", "I3", "Pj3"),
    _hub("seed4", f"{HUB_ORG}/pythia-160m-seed4", "I4", "Pj4"),
    BaseModel(
        "pile-ws1",
        "local",
        None,
        "I1",
        "P0",
        "pile",
        "V",
        "v1.0",
        WAVE_NOW,
        FINAL_STEP,
        "seed1",
    ),
    BaseModel(
        "dclm-main",
        "local",
        None,
        "I1234",
        "C",
        "dclm",
        "M",
        "modern",
        WAVE_NOW,
        FINAL_STEP,
        "pythia-160m",
    ),
    BaseModel(
        "dclm-ws1",
        "local",
        None,
        "I1",
        "C",
        "dclm",
        "M",
        "modern",
        WAVE_NOW,
        FINAL_STEP,
        "seed1",
    ),
    BaseModel(
        "pile-ws3",
        "local",
        None,
        "I3",
        "P0",
        "pile",
        "M",
        "modern",
        WAVE_2,
        FINAL_STEP,
        "seed3",
    ),
    BaseModel(
        "pile-ws2",
        "local",
        None,
        "I2",
        "P0",
        "pile",
        "V",
        "v1.0",
        WAVE_3,
        FINAL_STEP,
        "seed2",
    ),
    BaseModel(
        "pile-main",
        "local",
        None,
        "I1234",
        "P0",
        "pile",
        "M",
        "modern",
        WAVE_3,
        FINAL_STEP,
        "pythia-160m",
    ),
)
_BY_NAME: Mapping[str, BaseModel] = MappingProxyType({m.base: m for m in ROSTER})
# Goncalo's runs still training on 2026-09-18: documented as pending, never a roster entry.
PENDING_RUNS: frozenset[str] = frozenset(
    {"data-order-b", "data-order-b-v1", "pythia-160m-repro"}
)
# Never usable in any cell (doc § 4.6): uncontrolled Hub weight seeds, the stopped DCLM
# run, the never-fingerprinted seeds 5-9, and the pending runs.
EXCLUDED_BASES: frozenset[str] = frozenset(
    {"weight-seed1", "weight-seed2", "weight-seed3", "dclm-ws2"}
    | {f"seed{n}" for n in range(5, 10)}
    | PENDING_RUNS
)
# Doc § 5, one pair per category anchored on a local weight seed (or on the Hub
# original): teacher/student partners of the anchor, keyed by category.
MATCHED_SUBDESIGN: Mapping[str, Mapping[str, str | None]] = MappingProxyType(
    {
        "pile-ws1": MappingProxyType(
            {
                "SI_DS": "seed1",
                "DI_SS": "pythia-160m",
                "DI_DS": "seed2",
                "SI_DC": "dclm-ws1",
                "DI_DC": "dclm-main",
            }
        ),
        "pythia-160m": MappingProxyType(
            {
                "SI_DS": "data-seed1",
                "DI_DS": "seed1",
                "SI_DC": "dclm-main",
                "DI_DC": "dclm-ws1",
            }
        ),
        "pile-ws3": MappingProxyType(
            {
                "SI_DS": "seed3",
                "DI_SS": "pythia-160m",
                "DI_DS": "seed1",
                "SI_DC": None,
                "DI_DC": "dclm-main",
            }
        ),
        "pile-main": MappingProxyType(
            {
                "SI_DS": "data-seed1",
                "DI_SS": "pile-ws3",
                "DI_DS": "seed1",
                "SI_DC": "dclm-main",
                "DI_DC": "dclm-ws1",
            }
        ),
    }
)


def base(name: str) -> BaseModel:
    """Roster entry for an inventory name.

    Raises:
        ValueError: For an excluded or unknown base.
    """
    if name in EXCLUDED_BASES:
        raise ValueError(f"base {name!r} is excluded from every ws160m cell")
    try:
        return _BY_NAME[name]
    except KeyError as err:
        raise ValueError(f"unknown base {name!r}; roster: {sorted(_BY_NAME)}") from err


def bases(*, max_wave: int) -> tuple[BaseModel, ...]:
    """Roster entries usable at step143000 up to and including `max_wave`."""
    return tuple(m for m in ROSTER if m.wave <= max_wave)


def category(teacher: str, student: str) -> str:
    """Ablation category of the directed cell teacher-base -> student-base.

    Returns one of `CATEGORIES`, or `SI_SS_NONIDENT` for two distinct bases that share
    both init and stream.
    """
    t, s = base(teacher), base(student)
    if t.base == s.base:
        return "SI_SS"
    same_init, same_stream, same_corpus = (
        t.init == s.init,
        t.stream == s.stream,
        t.corpus == s.corpus,
    )
    if same_init and same_stream:
        return SI_SS_NONIDENT
    if same_init and same_corpus:
        return "SI_DS"
    if not same_init and same_stream:
        return "DI_SS"
    if not same_init and same_corpus:
        return "DI_DS"
    return "SI_DC" if same_init else "DI_DC"


def impl_stratum(teacher: str, student: str) -> str:
    """Implementation stratum of a pair (doc § 5)."""
    t, s = base(teacher), base(student)
    hubs = (t.impl == "H") + (s.impl == "H")
    if hubs == 2:
        return "hub_only"
    if hubs == 1:
        return "local_hub"
    return (
        "local_local_same_checkout"
        if t.checkout == s.checkout
        else "local_local_cross_stack"
    )


def init_stratum(teacher: str, student: str) -> str:
    """Shared init label of a same-init pair, else ``"mixed"``."""
    t, s = base(teacher), base(student)
    return t.init if t.init == s.init else "mixed"


def corpus_stratum(teacher: str, student: str) -> str:
    """``pile-pile``, ``dclm-dclm`` or ``pile-dclm`` (order-free)."""
    pair = sorted((base(teacher).corpus, base(student).corpus), reverse=True)
    return "-".join(pair)


def unordered_pairs(*, max_wave: int) -> tuple[tuple[str, str], ...]:
    """Every unordered pair of distinct usable bases, in roster order."""
    names = [m.base for m in bases(max_wave=max_wave)]
    return tuple(
        (names[i], names[j])
        for i in range(len(names))
        for j in range(i + 1, len(names))
    )


def pairs_by_category(*, max_wave: int) -> Mapping[str, tuple[tuple[str, str], ...]]:
    """Unordered pairs grouped by category; self-pairs listed as ``(b, b)`` under SI_SS."""
    out: dict[str, list[tuple[str, str]]] = {
        c: [] for c in (*CATEGORIES, SI_SS_NONIDENT)
    }
    for m in bases(max_wave=max_wave):
        out["SI_SS"].append((m.base, m.base))
    for a, b in unordered_pairs(max_wave=max_wave):
        out[category(a, b)].append((a, b))
    return MappingProxyType({k: tuple(v) for k, v in out.items()})


# ---- tokens, labels, specs --------------------------------------------------


def token(base_name: str, step: int) -> str:
    """Roster key / sweep token, ``<base>@step<N>``."""
    base(base_name)
    return f"{base_name}@step{int(step)}"


def parse_token(value: str) -> tuple[str, int]:
    """Inverse of `token`.

    Raises:
        ValueError: If the token is malformed or names an unknown base.
    """
    name, sep, rev = value.partition("@step")
    if not sep or not rev.isdigit():
        raise ValueError(f"malformed ws160m token {value!r}; expected <base>@step<N>")
    base(name)
    return name, int(rev)


def label(base_name: str, step: int) -> str:
    """Run-directory label, ``<base>-step<N>``."""
    base(base_name)
    return f"{base_name}-step{int(step)}"


def local_run_dir(base_models_root: Path, base_name: str) -> Path:
    """Staged NeoX run directory of a local base."""
    return Path(base_models_root) / base_name


def neox_revision_dir(base_models_root: Path, base_name: str, step: int) -> Path:
    """Staged NeoX ``global_step<N>`` directory of a local base."""
    return local_run_dir(base_models_root, base_name) / f"global_step{int(step)}"


def converted_root(base_models_root: Path, base_name: str) -> Path:
    """`HfModelSpec.local_root` of a local base (one ``step<N>/`` per revision)."""
    return Path(base_models_root) / HF_CONVERTED_DIRNAME / base_name


def converted_dir(base_models_root: Path, base_name: str, step: int) -> Path:
    """Converted HF snapshot directory of a local base at `step`."""
    return converted_root(base_models_root, base_name) / f"step{int(step)}"


def spec(base_name: str, step: int, *, base_models_root: Path) -> HfModelSpec:
    """Spec for one (base, step); local bases resolve under `base_models_root`.

    Availability of the step on disk is a staging fact checked by the staging
    report, not here (`ladder_cap` is informational).
    """
    m = base(base_name)
    common = dict(
        label=label(base_name, step),
        revision=f"step{int(step)}",
        family="pythia",
        cluster="neox",
        model_type="gpt_neox",
        vocab_size=VOCAB_SIZE,
        tokenizer_sha256=NEOX_TOKENIZER_SHA,
        lora_target_modules=NEOX_LORA,
    )
    if m.source == "hub":
        return HfModelSpec(repo_id=str(m.hub_repo), **common)
    return HfModelSpec(
        repo_id=f"local/{base_name}",
        local_root=converted_root(base_models_root, base_name),
        **common,
    )


def registry(
    entries: Iterable[tuple[str, int]], *, base_models_root: Path
) -> Mapping[str, HfModelSpec]:
    """Token -> spec mapping for the sweeps' ``--spec-registry ws160m`` lookup."""
    return MappingProxyType(
        {token(b, s): spec(b, s, base_models_root=base_models_root) for b, s in entries}
    )


def registry_for_paths(paths_config: Path | None) -> Mapping[str, HfModelSpec]:
    """The sweeps' ``--spec-registry ws160m`` roster, rooted by a paths YAML.

    Every roster base at every `CAMPAIGN_STEPS` step, with local bases under
    ``<audit_root>/base_models_160m``.

    Raises:
        ValueError: If `paths_config` is missing or has no ``audit_root``.
    """
    import yaml

    if paths_config is None:
        raise ValueError("--spec-registry ws160m needs --paths-config")
    doc = yaml.safe_load(Path(paths_config).read_text())
    audit_root = (doc or {}).get("audit_root")
    if not audit_root:
        raise ValueError(f"{paths_config}: audit_root is required for ws160m")
    root = Path(str(audit_root)) / BASE_MODELS_DIRNAME
    entries = [(m.base, step) for m in ROSTER for step in CAMPAIGN_STEPS]
    return registry(entries, base_models_root=root)
