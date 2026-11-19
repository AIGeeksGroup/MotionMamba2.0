"""Construct the configured frontend or the shared text encoder."""
import os
from motion_mamba2.model.configuration import load_environment,read_config

def load_text_encoder(mode=None,url=None,fp32=False,device=None):
    import torch
    mode=mode or os.environ.get('TEXT_ENCODER_MODE','api')
    if mode=='api':
        from motion_mamba2.model.text_encoder_api import TextEncoderAPI
        return TextEncoderAPI(url or os.environ.get('TEXT_ENCODER_URL','http://127.0.0.1:9550/')).to(device=device or 'cpu',dtype=torch.float32)
    if mode!='local':raise ValueError('Choose api or local text encoding')
    from motion_mamba2.model.language_wrapper import LLM2VecEncoder
    env=load_environment()
    for key in ['MM2_TEXT_ENCODER_BASE','MM2_TEXT_ENCODER_ADAPTER']:
        if not env.get(key):raise ValueError('Fill '+key+' in configs/paths.env')
    return LLM2VecEncoder(base_model_name_or_path=env['MM2_TEXT_ENCODER_BASE'],
        peft_model_name_or_path=env['MM2_TEXT_ENCODER_ADAPTER'],dtype='float32' if fp32 else 'bfloat16',
        llm_dim=4096,device=device or 'cuda:0')

def load_model(modelname=None,device='cuda:0',text_encoder=None,fresh_denoiser=False,**kwargs):
    from motion_mamba2.model.loading import instantiate_from_dict
    load_environment()
    config=read_config('configs/model.yaml')['frontend']
    m=instantiate_from_dict(config,overrides=dict(device=device,text_encoder=None))
    if text_encoder is not False:m.text_encoder=text_encoder or load_text_encoder(device=device)
    return m
