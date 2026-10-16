"""Frame-based source/time/visibility contract shared by GT and teacher labels."""
from dataclasses import dataclass, asdict

VERSION='g1_training_sample_v1'
@dataclass(frozen=True)
class TimeLayout:
    history:int
    future:int
    generation:int=52
    patch:int=4
    def validate(self):
        assert self.generation==52 and self.patch==4
        assert self.history>=0 and self.future>=0 and self.history+self.future<=196
        assert all(n%4==0 for n in [self.history,self.future,self.generation])
        return self
    @property
    def frames(self):return self.history+self.generation+self.future
    def dict(self):return asdict(self)
def validate_record(r):
    assert r['schema']==VERSION and r['source_family_id'] and r['split'] in ['TRAIN','DEV']
    assert r['fps']==25 and r['skeleton']=='g1' and r['representation']=='physical414'
    TimeLayout(r['H'],r['F']).validate()
    assert r['supervision'] in ['GT','same_state_teacher']
    if r['supervision']=='same_state_teacher':
        assert r['qualified'] and not r['teacher_extra_conditions'] and r['collector_checkpoint_sha256']
    return r
