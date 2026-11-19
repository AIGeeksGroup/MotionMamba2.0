"""Activate public source and caller-supplied assets, never experiment directories."""
import os
import sys
from pathlib import Path
SIM=Path(__file__).resolve().parents[1]
LIVE=SIM/'inference'
SOURCE=SIM.parent/'motion_mamba2'

def activate():
    for p in [SIM.parent,SIM/'vendor/viser',LIVE,SIM/'viewer',SIM/'web/backend']:
        if str(p) not in sys.path:sys.path.insert(0,str(p))
    import motion_mamba2
    import motion_mamba2.model.assets as assets
    asset_root=os.environ.get('MM2_VIEWER_ASSETS','')
    if asset_root:
        assets.ASSETS_ROOT=Path(asset_root).expanduser().resolve()
        assets.SKELETONS_ROOT=assets.ASSETS_ROOT/'skeletons'
        assets.DEMO_ASSETS_ROOT=assets.ASSETS_ROOT/'demo'
        assets.DEMO_EXAMPLES_ROOT=assets.DEMO_ASSETS_ROOT/'examples'
        assets.SOMA_ASSETS_ROOT=assets.ASSETS_ROOT/'SOMA'
    return SIM

_text={}
def warm_embedding(device):
    """Obtain deterministic real text features through the configured service."""
    import torch
    from motion_mamba2.model.text_encoder_api import TextEncoderAPI
    url=os.environ.get('TEXT_ENCODER_URL','http://127.0.0.1:9550/')
    if url not in _text:
        encoder=TextEncoderAPI(url).to(device='cpu',dtype=torch.float32)
        feature,length=encoder(['Wave with the right hand naturally.'])
        if tuple(feature.shape)!=(1,1,4096) or length!=[1]:raise ValueError('Unexpected LLM2Vec feature shape')
        _text[url]=feature
    return _text[url].to(device=device,dtype=torch.float32)
