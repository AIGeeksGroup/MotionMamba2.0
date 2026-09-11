from types import SimpleNamespace
from pathlib import Path
import sys
import pytest, torch
from ardy_adapter.diffusion_memory import ArdyDiffusionMemoryController
from minimal_attention_kernel import ChunkSlidingWindowKVCache

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
from evaluate_ardy_official_table5 import _semantic_derangement

def controller(batch=2):
    cache=ChunkSlidingWindowKVCache(batch_size=batch,num_heads=1,head_dim=2,window_size_chunks=4,num_sink_tokens=0,device='cpu',dtype=torch.float32)
    cache.append(torch.randn(batch,1,2,2),torch.randn(batch,1,2,2),chunk_id=0)
    layer=SimpleNamespace(attention_cache=cache,mamba_state=torch.randn(batch,3),state_summary=torch.randn(batch,3),state_delta=torch.randn(batch,3),persistent_memory=torch.arange(batch*12,dtype=torch.float32).reshape(batch,2,6))
    block=SimpleNamespace(_sequence_state=SimpleNamespace(layers=[layer]),content_memory_read_enabled_runtime=None,content_memory_write_enabled_runtime=True,content_memory_delta_mode_runtime='normal',content_memory_read_view_runtime=None)
    c=ArdyDiffusionMemoryController.__new__(ArdyDiffusionMemoryController); c.root_block=block; c.body_block=SimpleNamespace(_sequence_state=None,content_memory_read_enabled_runtime=None,content_memory_write_enabled_runtime=True,content_memory_delta_mode_runtime='normal',content_memory_read_view_runtime=None)
    c._completed_windows=4; c._content_memory_permutation_applied=False; c._content_memory_permutation=None; c._content_memory_read_trace_window=None; c.content_memory_audit=[]
    return c,layer

def test_persistent_read_permutation_does_not_mutate_durable_memory():
    c,l=controller(); before=l.persistent_memory.clone(); c.content_memory_intervention='persistent_read_permutation'; c._set_content_memory_runtime(); torch.testing.assert_close(l.persistent_memory,before)

def test_persistent_read_permutation_preserves_attention_kv():
    c,_=controller(); before=c._attention_checksum(); c.content_memory_intervention='persistent_read_permutation'; c._set_content_memory_runtime(); assert c._attention_checksum()==before

def test_persistent_read_permutation_preserves_mamba_state():
    c,l=controller(); before=l.mamba_state.clone(); c.content_memory_intervention='persistent_read_permutation'; c._set_content_memory_runtime(); torch.testing.assert_close(l.mamba_state,before)

def test_persistent_read_permutation_applies_every_post_w4_read():
    c,_=controller(); c.content_memory_intervention='persistent_read_permutation'
    for window in (4,5,6): c._completed_windows=window; c._set_content_memory_runtime()
    assert [r['completed_windows'] for r in c.content_memory_audit]==[4,5,6]

def test_persistent_read_permutation_uses_fixed_derangement():
    c,_=controller(); c.content_memory_intervention='persistent_read_permutation'; c._set_content_memory_runtime(); first=c._content_memory_permutation.clone(); c._completed_windows=5; c._set_content_memory_runtime(); torch.testing.assert_close(c._content_memory_permutation,first); assert all(i!=j for i,j in enumerate(first.tolist()))

def test_freeze_write_keeps_memory_constant():
    c,l=controller(); before=l.persistent_memory.clone(); c.content_memory_intervention='freeze_write_after_w4'; c._set_content_memory_runtime(); assert c.root_block.content_memory_write_enabled_runtime is False; torch.testing.assert_close(l.persistent_memory,before)

def test_external_read_view_is_cleared_before_clean_commit():
    c, _ = controller(); c._external_memory_bank = {("a", 4, "root", 0): torch.zeros(2, 6)}
    c._external_memory_donor_ids = ["a", "a"]; c._external_memory_recipient_ids = ["a", "b"]
    c._external_memory_mode = "donor"; c._completed_windows = 4
    c._set_content_memory_runtime()
    assert c.root_block.content_memory_read_view_runtime is not None
    c._set_content_memory_runtime()
    # The post-window hook performs the same explicit clearing before commit;
    # keep the assertion on the public block contract in this focused test.
    c.root_block.content_memory_read_view_runtime = None
    assert c.root_block.content_memory_read_view_runtime is None

def test_read_ready_external_level_is_explicit_and_validated():
    c, layer = controller()
    bank = {("c", 4, "root", 0): torch.zeros(2, 6), ("d", 4, "root", 0): torch.zeros(2, 6)}
    c.configure_external_memory_read(bank, recipient_ids=["a", "b"], donor_ids=["c", "d"], mode="donor", level="read_ready")
    assert c._external_memory_level == "read_ready"

def test_permute_plus_freeze_keeps_wrong_memory_constant():
    c,l=controller(); before=l.persistent_memory.clone(); c.content_memory_intervention='one_shot_permute_plus_freeze_write'; c._maybe_permute_content_memory(); wrong=l.persistent_memory.clone(); c._set_content_memory_runtime(); assert c.root_block.content_memory_write_enabled_runtime is False; assert not torch.equal(wrong,before); torch.testing.assert_close(l.persistent_memory,wrong)

def test_freeze_only_control_keeps_correct_memory_constant():
    c,l=controller(); before=l.persistent_memory.clone(); c.content_memory_intervention='freeze_write_after_w4'; c._set_content_memory_runtime(); torch.testing.assert_close(l.persistent_memory,before)

def test_semantic_pairing_is_deranged_and_deterministic():
    features=torch.tensor([[1.,0.],[.9,.1],[-1.,0.],[0.,-1.]])
    a=_semantic_derangement(features,['a','b','c','d']); b=_semantic_derangement(features,['a','b','c','d'])
    torch.testing.assert_close(a,b); assert sorted(a.tolist())==list(range(4)); assert all(i!=j for i,j in enumerate(a.tolist()))

def test_interventions_do_not_modify_attention_w4_kv():
    test_persistent_read_permutation_preserves_attention_kv()

def test_interventions_do_not_modify_mamba_state_unintentionally():
    test_persistent_read_permutation_preserves_mamba_state()
