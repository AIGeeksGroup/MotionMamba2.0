import re
IDENTITY_FIELDS=('source_family_id','source_motion_id','source_id','sample_id','canonical_take','sealed_native_sha256','actor_id_namespaced')

def caption_task_tags(caption):
    patterns={'walk':r'\bwalk(?:s|ing)?\b','run':r'\b(?:run|runs|running|jog|jogging)\b',
              'turn':r'\bturn(?:s|ing)?\b','wave':r'\bwav(?:e|es|ing)\b',
              'squat':r'\bsquat(?:s|ting)?\b','bend':r'\bbend(?:s|ing)?\b',
              'kick':r'\bkick(?:s|ing)?\b','jump':r'\bjump(?:s|ing)?\b',
              'dance':r'\bdanc(?:e|es|ing)\b','clap':r'\bclap(?:s|ping)?\b'}
    return [k for k,p in patterns.items() if re.search(p,caption.lower())] or ['unknown']


def excluded(row, registry):
    for k in IDENTITY_FIELDS:
        values=registry.get(k+'s', registry.get(k, []))
        if row.get(k) is not None and row[k] in values:return True
    return False

