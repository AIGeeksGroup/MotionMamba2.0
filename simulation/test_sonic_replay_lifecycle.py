"""Real bounded bridge regression: evicted replay, owner controls, reconnect."""
import importlib.util
import os
from pathlib import Path
import socket
import sys
import time
import unittest
from unittest.mock import patch

import numpy as np
import zmq

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'ardy/scripts'))
from interactive_demo import sonic_output_bridge as bridge_module

def port():
    with socket.socket() as s:
        s.bind(('127.0.0.1',0));return s.getsockname()[1]

def wait(predicate,timeout=5):
    end=time.monotonic()+timeout
    while time.monotonic()<end:
        if predicate():return
        time.sleep(.02)
    raise AssertionError('Timed out')

def motion(frames):
    q=np.zeros((frames,36));q[:,2]=.8;q[:,3]=1
    q[:,0]=np.arange(frames)*.002
    return q

class Replay(unittest.TestCase):
    def test_restart_reloads_evicted_prefix_then_accepts_the_next_window(self):
        b=bridge_module.SonicOutputBridge(web_url=None,zmq_bind=None)
        try:
            q=motion(52*8)
            for i in range(8):
                with b._lock:b._state.frame=max(0,i*104-60)
                b.append_source_window(q[i*52:(i+1)*52],source_start=i*52,name='walk')
            self.assertGreater(b._state.frame_origin,0)
            def control(action):
                self.assertEqual(action,'restart')
                b.restore_source_playback(q,source_frame=0,name='walk')
                return True
            b.source_control=control
            b.viewer_playback_control('restart')
            self.assertEqual(b._state.frame,0)
            self.assertEqual(b._state.frame_origin,0)
            np.testing.assert_array_equal(b._state.qpos[0],q[0].astype(np.float32))
            b.append_source_window(motion(52),source_start=416,name='walk')
            wait(lambda:b._thread.is_alive())
            self.assertIsNone(b.stream_error)
            self.assertEqual(b._state.source_next,468)
        finally:b.close()

    def test_orphaned_evicted_cache_does_not_build_negative_slices(self):
        b=bridge_module.SonicOutputBridge(web_url=None,zmq_bind=None)
        try:
            b.restore_source_playback(motion(416),source_frame=300,name='walk')
            b.viewer_playback_control('restart')
            self.assertEqual(b._state.frame,600)
            self.assertFalse(b._state.playing)
            self.assertTrue(b._thread.is_alive())
        finally:b.close()

    def test_packet_failure_reconnects_paused_and_sonic_controls_reach_owner(self):
        original=bridge_module.ZmqPublisher
        attempts=[];calls=[]
        class Publisher(original):
            def send_pose(self,**kwargs):
                attempts.append(kwargs['frame_index'].copy())
                if len(attempts)==1:raise RuntimeError('injected send failure')
                return super().send_pose(**kwargs)
        control_port=port();pose_port=port()
        with patch.dict(os.environ,{'ARDY_SONIC_CONTROL_BIND':f'tcp://127.0.0.1:{control_port}',
                'ARDY_SONIC_MODE_SWITCH_SECONDS':'0','ARDY_SONIC_REFERENCE_WARMUP_SECONDS':'0'}),patch.object(bridge_module,'ZmqPublisher',Publisher):
            b=bridge_module.SonicOutputBridge(web_url=None,zmq_bind=f'tcp://127.0.0.1:{pose_port}')
            ctx=zmq.Context();sender=ctx.socket(zmq.PUSH);sender.setsockopt(zmq.LINGER,0)
            sender.connect(f'tcp://127.0.0.1:{control_port}')
            try:
                b.append_source_window(motion(52),source_start=0,name='walk')
                wait(lambda:len(attempts)>=2)
                self.assertTrue(b._thread.is_alive());self.assertFalse(b._state.playing)
                self.assertIsNone(b.stream_error)
                def owner(action):
                    if action=='restart':b.restore_source_playback(motion(52),source_frame=0,name='walk')
                    else:b.viewer_playback_control(action)
                    calls.append(action)
                    return True
                b.source_control=owner
                for action in ('play','pause','restart'):
                    sender.send_json({'action':action})
                    wait(lambda:action in calls)
                self.assertEqual(calls,['play','pause','restart'])
                self.assertTrue(b._thread.is_alive())
                self.assertEqual(b._state.frame,0)
            finally:
                b.close();sender.close();ctx.term()

if __name__=='__main__':unittest.main()
