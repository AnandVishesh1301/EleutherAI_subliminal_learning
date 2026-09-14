"""CPU-only, offline preflight for executing the unchanged frozen corrections."""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import platform
import sys


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check_hashes(root, hashes):
    if not hashes:
        raise ValueError("empty provenance inventory")
    for name, digest in hashes.items():
        path = root / name
        if sha256(path) != digest:
            raise ValueError(f"hash mismatch: {path}")


def check_weights(snapshot):
    for index in ("model.safetensors.index.json", "pytorch_model.bin.index.json"):
        path = snapshot / index
        if path.is_file():
            names = set(json.loads(path.read_text())["weight_map"].values())
            break
    else:
        names = {n for n in ("model.safetensors", "pytorch_model.bin")
                 if (snapshot / n).is_file()}
    if not names:
        raise ValueError(f"missing model weights: {snapshot}")
    for name in names:
        path = snapshot / name
        if not path.is_file() or not os.access(path, os.R_OK):
            raise ValueError(f"unreadable model shard: {path}")
    return len(names)


def preflight(campaign):
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    campaign = campaign.resolve()
    frozen = campaign / "code"
    m = json.loads((campaign / "manifest.json").read_text())
    check_hashes(frozen, m["code_sha256"])
    check_hashes(campaign, m["file_sha256"])
    for rec in m["inputs"].values():
        path = Path(rec["weights_path"])
        st = path.stat()
        if (st.st_size, st.st_mtime_ns) != (rec["weights_size"], rec["weights_mtime_ns"]):
            raise ValueError(f"changed checkpoint: {path}")
        if not os.access(path, os.R_OK):
            raise ValueError(f"unreadable checkpoint: {path}")
        for name, digest in rec["metadata_sha256"].items():
            if sha256(name) != digest:
                raise ValueError(f"metadata mismatch: {name}")
    out = Path(m["output_root"])
    for path in (out, out / "analysis", out / "analysis" / campaign.name / "olmo"):
        if not os.access(path, os.R_OK | os.W_OK | os.X_OK):
            raise ValueError(f"output directory inaccessible: {path}")
    os.chdir(frozen)
    sys.path[:0] = [str(frozen / "src"), str(frozen)]
    for name in ("scripts.position_surgery", "subliminal_learning.models.loading_io"):
        module = importlib.import_module(name)
        if not Path(module.__file__).resolve().is_relative_to(frozen):
            raise ValueError(f"import escaped frozen source: {module.__file__}")
    import torch
    import transformers
    import yaml
    from huggingface_hub import try_to_load_from_cache
    from subliminal_learning.models.datadecide import load_datadecide_tokenizer, parse_dd_ref
    from subliminal_learning.coherence.probe_io import build_probe_pool, PROBE_POOL_DP1

    paths = yaml.safe_load((frozen / "configs/paths/paths_local.yaml").read_text())
    cache = Path(paths["hf_cache_root"])
    universe = Path(paths["dataset_cache_root"]) / "dolma_dp1/dolma_v1_7_universe.jsonl.gz"
    if not universe.is_file():
        raise ValueError(f"missing offline probe universe: {universe}")
    models = []
    for ref in sorted({rec["base_ref"] for rec in m["inputs"].values()}):
        repo, revision = ref.split("@", 1)
        config = try_to_load_from_cache(repo, "config.json", revision=revision,
                                        cache_dir=str(cache / "hub"))
        if not isinstance(config, str):
            raise ValueError(f"missing cached config: {ref}")
        shards = check_weights(Path(config).parent)
        tok = load_datadecide_tokenizer(parse_dd_ref(ref), cache)
        pool = build_probe_pool(tok, PROBE_POOL_DP1)
        models.append(dict(ref=ref, shards=shards, pool_sha=pool.token_sha256()))
    if torch.cuda.is_initialized():
        raise ValueError("CPU preflight unexpectedly initialized CUDA")
    return dict(code_files=len(m["code_sha256"]), inputs=len(m["inputs"]),
                torch=torch.__version__, transformers=transformers.__version__, models=models)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    args = parser.parse_args()
    report = dict(hostname=platform.node(), python=platform.python_version(), ok=False)
    try:
        report.update(preflight(args.campaign), ok=True)
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
    print(json.dumps(report, sort_keys=True), flush=True)
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
