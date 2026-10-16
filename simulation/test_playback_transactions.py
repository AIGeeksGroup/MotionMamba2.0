"""Deterministic regression checks of real frontend methods, without services."""
import ast
import json
from pathlib import Path
import tempfile
import threading
import time
from types import SimpleNamespace as NS
import unittest
import sys

import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'ardy/scripts'))
from interactive_demo.control_history import ControlHistory

SOURCE=Path(__file__).resolve().parents[1]/'ardy/scripts/interactive_demo/motion_mamba_backend.py'
tree=ast.parse(SOURCE.read_text())
cls=next(n for n in tree.body if isinstance(n,ast.ClassDef))
names={'write_trace','reset','_reset_impl','seek_playback','generate','source_playback_control'}
nodes=[n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name in names]
generation=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='_generate_impl')
loop=next(n for n in ast.walk(generation) if isinstance(n,(ast.For,ast.With)) and any(
    isinstance(v,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='start' for t in v.targets)
    for v in n.body))
start=next(i for i,n in enumerate(loop.body) if isinstance(n,ast.Assign) and any(
    isinstance(t,ast.Name) and t.id=='start' for t in n.targets))
end=next((i for i,n in enumerate(loop.body) if isinstance(n,ast.Assign) and any(
    isinstance(t,ast.Name) and t.id=='record' for t in n.targets)),len(loop.body))
nodes.append(ast.FunctionDef(name='commit_window',args=ast.arguments(posonlyargs=[],args=[
    ast.arg(arg=n) for n in ('self','demo','session','converted','prompt','transitions')],kwonlyargs=[],kw_defaults=[],defaults=[]),
    body=loop.body[start:end],decorator_list=[]))
namespace=dict(threading=threading,time=time,np=np,json=json,Path=Path,os=__import__('os'),ControlHistory=ControlHistory)
exec(compile(ast.fix_missing_locations(ast.Module(body=[ast.ClassDef(name='Methods',bases=[],keywords=[],body=nodes,decorator_list=[])],type_ignores=[])),str(SOURCE),'exec'),namespace)
Methods=namespace['Methods']


class Transactions(unittest.TestCase):
    def test_sonic_restart_resets_viewer_anchor_before_render_only_refresh(self):
        m=Methods();m.motion_update_lock=threading.RLock();m.generation_lock=threading.RLock();m.sid='live';m.window=8
        s=NS(client=NS(client_id=0),frame_float=372.5,frame_idx=372)
        d=NS(client_active=lambda _:True,mm2_physical_owner=0,
             set_playback_state=lambda *args:None)
        def seek(*args,**kwargs):self.assertTrue(kwargs['force']);self.assertEqual(args[2],0)
        m.seek_playback=seek
        def anchor(session,frame):session.frame_float=frame;session.frame_idx=int(frame)
        d._anchor_playback=anchor
        def render(client_id,frame,**kwargs):
            # Actual render_only intentionally preserves the existing subframe.
            self.assertTrue(kwargs['render_only']);self.assertEqual(s.frame_float,0)
        d.set_frame=render
        self.assertTrue(m.source_playback_control(d,s,'restart'))
        self.assertEqual(m.consumed_frame,0)

    def test_reset_directory_is_ready_before_logger_can_observe_it(self):
        with tempfile.TemporaryDirectory() as folder:
            m=Methods();m.output=Path(folder);m.motion_update_lock=threading.RLock();m.generation_lock=threading.RLock()
            m.run_dir=m.output/'old';m.run_dir.mkdir();m.sid='old'
            m.close=lambda:None;m.reference_clock=NS(reset=lambda:None)
            m.api=lambda *args:dict(session='new')
            session=NS(gui_elements=NS(gui_seed=NS(value=2),gui_prompt_text=NS(value='walk')),
                       init_global_translation=np.zeros(3),init_first_heading_angle=0.)
            m.limit=0
            original=Path.mkdir
            def inspect_before_mkdir(path,*args,**kwargs):
                if path.name=='new':
                    # Force the precise failing interleaving from the production trace.
                    self.assertEqual(m.run_dir.name,'old')
                    m.write_trace('playback.jsonl',{'session':m.sid})
                return original(path,*args,**kwargs)
            from unittest.mock import patch
            with patch.object(Path,'mkdir',inspect_before_mkdir):m.reset(session)
            self.assertEqual(m.sid,'new');self.assertTrue(m.run_dir.is_dir())
            m.write_trace('playback.jsonl',{'session':m.sid})
            self.assertTrue((m.run_dir/'playback.jsonl').exists())

    def test_log_io_failure_does_not_escape_to_playback_thread(self):
        m=Methods();m.run_dir=Path('/definitely-missing-mm2-test-folder')
        m.write_trace('playback.jsonl',{'playing':True})

    def test_seek_waits_until_generation_has_committed_both_frontiers(self):
        m=Methods();m.motion_update_lock=threading.RLock();m.generation_lock=threading.RLock()
        entered=threading.Event();release=threading.Event();seek_done=threading.Event()
        m.frontier=520;m.bridge_frontier=520
        def generate(*args):
            m.frontier=572;entered.set();release.wait(2);m.bridge_frontier=572
        def seek(*args,**kwargs):
            self.assertEqual(m.frontier,m.bridge_frontier);seek_done.set()
        m._generate_impl=generate;m._seek_playback_impl=seek
        a=threading.Thread(target=m.generate,args=(None,None));a.start();self.assertTrue(entered.wait(1))
        b=threading.Thread(target=m.seek_playback,args=(None,None,0));b.start()
        self.assertFalse(seek_done.wait(.05));release.set();a.join();b.join();self.assertTrue(seek_done.is_set())

    def test_timeline_callback_cannot_publish_a_window_before_append(self):
        m=Methods();m.window=10;m.limit=0;m.segments=[];m.reference_preview=False
        m.consumed_frame=480;m.prompt_at=lambda frame:'walk'
        state=NS(source_next=520)
        def append(converted,*,source_start,**kw):
            # This raised 520 != 572 in the failing production execution.
            self.assertEqual(source_start,state.source_next);state.source_next+=52
        bridge=NS(append_source_window=append)
        session=NS(max_frame_idx=571,gui_elements=NS(gui_frame_idx_input=NS(max=519)))
        def callback(*args):
            self.assertEqual(state.source_next,m.window*52)
            state.source_next=m.window*52  # simulate restore_source_playback
        m.sync_timeline=callback
        m.commit_window(NS(sonic_output=bridge),session,None,'walk',[])
        self.assertEqual(state.source_next,572)


if __name__=='__main__':unittest.main()
