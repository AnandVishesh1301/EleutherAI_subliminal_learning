"""Small in-memory replication of the OLMo injection CLI's pool and dose contract."""
import json
from types import SimpleNamespace

import numpy as np

from scripts import diffuse_ladder_surgery as diffuse


def test_diffuse_uses_dp1_and_keeps_targets_fixed_across_doses(tmp_path, monkeypatch):
    calls=[]
    monkeypatch.setattr('subliminal_learning.config.resolve_roots',lambda *a:(tmp_path,tmp_path))
    monkeypatch.setattr('subliminal_learning.models.loading_io.load_base_from_ref',
                        lambda *a:(object(),object()))
    def build(tok,spec):
        assert spec.version=='dp1'
        return SimpleNamespace(token_sha256=lambda:'dp1-sha')
    monkeypatch.setattr('subliminal_learning.coherence.probe_io.resolve_probe_pool_spec',
                        lambda repo,pool=None:SimpleNamespace(version=pool))
    monkeypatch.setattr('subliminal_learning.coherence.probe_io.build_probe_pool',build)
    monkeypatch.setattr('subliminal_learning.coherence.probe_io.parked_logprobs',lambda *a:[])
    monkeypatch.setattr('subliminal_learning.coherence.probe_io.mean_ce',lambda *a:2.)
    mods={name:dict(shape=(4,4),U=np.eye(4),Vt=np.eye(4),S=np.ones(4),W0np=np.eye(4))
          for name in ['m1','m2']}
    monkeypatch.setattr('subliminal_learning.weightspace.patching.base_param_svds',lambda *a,**k:mods)
    monkeypatch.setattr('subliminal_learning.analysis.intruder_threshold.theta_bar',lambda *a:1.)
    monkeypatch.setattr('subliminal_learning.weightspace.patching.patch_param_modules',
                        lambda m,d:calls.append({k:v.copy() for k,v in d.items()}))
    monkeypatch.setattr('subliminal_learning.weightspace.patching.restore_param_modules',lambda *a:None)
    def readout(*a,**kw):
        assert kw['logp_device']=='cpu'
        return 1.,1.
    monkeypatch.setattr('subliminal_learning.weightspace.patching.pile_readout',readout)
    monkeypatch.setattr('subliminal_learning.weightspace.deltas.intruder_positive',lambda *a:False)
    diffuse.main(repo='allenai/test@seed',out_name='tiny',k_ladder='1,2',e_ladder='.5,1',
                 seed=0,intruder_stride=1,match='fro',device='cpu',pool='dp1',
                 logp_device='cpu',output_root=tmp_path)
    for i in range(0,len(calls),2):
        for name in mods:
            np.testing.assert_allclose(2*calls[i][name],calls[i+1][name])
            np.testing.assert_allclose(np.linalg.norm(calls[i][name]),.5)
    assert not np.allclose(calls[0]['m1'],calls[0]['m2'])
    out=json.loads((tmp_path/'analysis/tiny.json').read_text())
    assert out['pool_version']=='dp1' and out['pool_sha']=='dp1-sha'
    assert out['repo']=='allenai/test@seed' and len(out['results'])==8
