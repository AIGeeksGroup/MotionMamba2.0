"""Architecture selection shared by training and benchmark inference."""

def add_model_flags(parser):
    group = parser.add_mutually_exclusive_group()
    parser.set_defaults(minimal_baseline=False)
    group.add_argument('--minimal_baseline', action='store_true',
                       help='Use softmax attention with sliding KV and learned sinks.')
    group.add_argument('--mamba_transformer_pcm', action='store_false', dest='minimal_baseline',
                       help='Use the released attention–Mamba model with two PCM slots (default).')


def read_model_config(path, *, minimal_baseline=False):
    from .configuration import read_config

    spec = read_config(path)
    if minimal_baseline:
        spec['sequence'].update(mode='minimal', token_state_interaction='naive_gated_add',
                                content_memory_enabled=False, content_memory_read_enabled=False,
                                content_memory_require_write=False)
    return spec
