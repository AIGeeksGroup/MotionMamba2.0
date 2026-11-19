"""Motion Mamba 2.0 architecture; imports do not load model assets."""
from importlib import import_module
_EXPORTS={
 'MotionMamba2':('motion_model','MotionMamba2'),
 'LLM2VecEncoder':('language_wrapper','LLM2VecEncoder'),
 'LLM2Vec':('language_encoder','LLM2Vec'),
 'load_model':('motion_mamba2.model.frontend','load_model'),
 'MotionMamba2MotionRep':('representation','MotionMamba2MotionRep'),
 'MotionRepBase':('representation_base','MotionRepBase'),
 'SequenceModelConfig':('integration','SequenceModelConfig'),
 'install_motion_mamba2_sequence_kernel':('integration','install_motion_mamba2_sequence_kernel'),
}
def __getattr__(name):
    if name not in _EXPORTS:raise AttributeError(name)
    module,attribute=_EXPORTS[name]
    if not module.startswith('motion_mamba2.'):module=__name__+'.'+module
    value=getattr(import_module(module),attribute)
    globals()[name]=value
    return value
