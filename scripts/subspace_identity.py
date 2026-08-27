"""Subspace-identity: do runs of one arm learn a SHARED carrier subspace, or merely
equally-toxic but disjoint directions? Metric per module: mean squared singular value
of U_a^T U_b between two runs' top-k left update subspaces (1 = identical, k/n ~ 0.005
= random). Averaged over modules; pairs within-arm vs cross-arm vs random baseline."""
import json, numpy as np
from pathlib import Path
from safetensors import safe_open
rng = np.random.default_rng(3)
K = 4
def topk(run):
    out = {}
    with safe_open(str(Path(run)/'checkpoints/adapter_model.safetensors'), 'np') as sf:
        tens = {k: sf.get_tensor(k) for k in sf.keys() if '.lora_' in k}
    for ka in [k for k in tens if k.endswith('lora_A.weight')]:
        mod = ka[len('base_model.model.'):-len('.lora_A.weight')]
        A = tens[ka].astype(np.float64); B = tens[ka.replace('lora_A','lora_B')].astype(np.float64)
        if min(B.shape[1], A.shape[0]) < K: return None
        qb, rb = np.linalg.qr(B); qa, ra = np.linalg.qr(A.T)
        uc = np.linalg.svd(rb @ ra.T)[0]
        out[mod] = (qb @ uc)[:, :K]
    return out
def overlap(sa, sb):
    vals = []
    for mod in sa:
        if mod not in sb: continue
        sv = np.linalg.svd(sa[mod].T @ sb[mod], compute_uv=False)
        vals.append(float(np.mean(sv**2)))
    return float(np.mean(vals))
strip = [l for l in Path('run_logs/harvest_20260824/lambda_arms_runs.txt').read_text().split() if '/pythia-160m-data-seed1/' in l and '/controls/' not in l][:10]
tf = [l for l in Path('run_logs/harvest_20260824/lambda_arms_runs.txt').read_text().split() if '/controls/' in l][:10]
Ss = [topk(r) for r in strip]; Ss = [s for s in Ss if s]
St = [topk(r) for r in tf]; St = [s for s in St if s]
def pairs(A_, B_=None):
    out = []
    if B_ is None:
        for i in range(len(A_)):
            for j in range(i+1, len(A_)): out.append(overlap(A_[i], A_[j]))
    else:
        for a in A_:
            for b in B_: out.append(overlap(a, b))
    return np.array(out)
w_strip = pairs(Ss); w_tf = pairs(St); x_arm = pairs(Ss, St)
n = 768
rand_ref = K / n  # expected mean sq cosine for random k-dim subspaces in R^n (approx k/n per direction)
print(f"mean top-{K} subspace overlap (1=identical, random ~{rand_ref:.4f}):")
print(f"  within strip arm    : {w_strip.mean():.4f} (n={len(w_strip)} pairs, sd {w_strip.std():.4f})")
print(f"  within trait-free   : {w_tf.mean():.4f} (n={len(w_tf)})")
print(f"  strip vs trait-free : {x_arm.mean():.4f} (n={len(x_arm)})")
Path('/mnt/cold-1/soar-subliminal/angela/outputs/analysis/subspace_identity_topk.json').write_text(json.dumps(dict(
    K=K, n_strip=len(Ss), n_tf=len(St), within_strip=float(w_strip.mean()), within_tf=float(w_tf.mean()),
    cross=float(x_arm.mean()), rand_ref=rand_ref, sd_within_strip=float(w_strip.std()))))
print('saved -> outputs/analysis/subspace_identity_topk.json')
