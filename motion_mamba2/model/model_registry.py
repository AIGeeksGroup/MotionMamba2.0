"""Single public G1 model selection for the interactive frontend."""
DEFAULT_HORIZON={'g1':52}
MODELS_BY_SKELETON={'g1':{52:'MotionMamba2'}}
MODELS={'g1':'MotionMamba2','g152':'MotionMamba2'}
DEFAULT_MODEL='MotionMamba2'
DEFAULT_TEXT_ENCODER_URL='http://127.0.0.1:9550/'
MODEL_NAMES={name:'AIGeeksGroup/MotionMamba2.0' for name in MODELS}
AVAILABLE_MODELS=['MotionMamba2']
def parse_model_name(name):return ('g1',52)
def resolve_model_name(name,default_family=None,checkpoints_dir=None):
    if name not in ['MotionMamba2',*MODELS]:raise ValueError('Only MotionMamba2 is available')
    return 'MotionMamba2'
def hf_repo_id(name):return 'AIGeeksGroup/MotionMamba2.0'
