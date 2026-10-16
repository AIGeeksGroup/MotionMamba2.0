"""Replay accepted directions without turning a seek into a new control input."""
import ast
from pathlib import Path
from types import SimpleNamespace as NS
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'ardy/scripts'))
from interactive_demo.control_history import ControlHistory

ROOT = Path(__file__).resolve().parents[1]/'ardy/scripts/interactive_demo'


def methods(filename, names):
    tree = ast.parse((ROOT / filename).read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef))
    nodes = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in names]
    namespace = dict(np=np)
    module = ast.Module(body=[ast.ClassDef(name='Methods', bases=[], keywords=[],
                                         body=nodes, decorator_list=[])], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(ROOT / filename), 'exec'), namespace)
    return namespace['Methods']


Model = methods('motion_mamba_backend.py', {'record_control', 'target_velocity_at', 'display_target_velocity_at', '_seek_playback_impl'})
Camera = methods('camera.py', {'on_arrow_key_press'})

# Exercise the actual rendering block, without requiring a GPU or a browser.
tree = ast.parse((ROOT / 'playback.py').read_text())
set_frame = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == 'set_frame')
arrow = next(n for n in set_frame.body if isinstance(n, ast.If)
             and ast.unparse(n.test) == 'session.target_velocity_arrow is not None')
function = ast.FunctionDef(name='render', args=ast.arguments(posonlyargs=[], args=[
    ast.arg(arg=n) for n in ('session', 'render_frame_idx', 'root_pos_for_target', 'preview_target_velocity')],
    kwonlyargs=[], kw_defaults=[], defaults=[ast.Constant(value=False)]), body=[arrow], decorator_list=[])
namespace = dict(np=np, TARGET_VELOCITY_UPDATE_INTERVAL=4, client_id=1,
                 self=NS(_update_root_constraints_from_target_velocity=lambda *args: None,
                         _clear_target_velocity_waypoints=lambda *args: None))
exec(compile(ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[])),
             str(ROOT / 'playback.py'), 'exec'), namespace)
render = namespace['render']


class Replay(unittest.TestCase):
    def setup_case(self):
        model = Model()
        model.is_remote_mm2 = True
        model.window = 4
        model.control_history = ControlHistory()
        self.events = []
        model.event = lambda name, **kw: self.events.append((name, kw))
        model.record_control(0, {'velocity_xz': [0, .6]})
        model.record_control(52, {'velocity_xz': [.6, 0]})
        model.record_control(104, {'velocity_xz': [0, -.6]})
        self.drawn = []
        g = NS(gui_use_target_velocity_checkbox=NS(value=True),
               gui_target_root_velocity=NS(value=(0, -.6)))
        s = NS(model=model, gui_elements=g, frame_idx=0,
               target_velocity_arrow=NS(update=lambda **kw: self.drawn.append(kw),
                                        set_visibility=lambda visible: self.drawn.append({'visible': visible})))
        return model, s

    def test_rewind_and_replay_show_each_accepted_direction_without_changing_input(self):
        m, s = self.setup_case()
        for frame, expected in [(150, [0, 0, -.6]), (0, [0, 0, .6]),
                                (51, [0, 0, .6]), (52, [.6, 0, 0]), (104, [0, 0, -.6])]:
            render(s, frame, np.zeros(3))
            np.testing.assert_allclose(self.drawn[-1]['root_velocity'], expected)
        self.assertEqual(s.gui_elements.gui_target_root_velocity.value, (0, -.6))
        self.assertEqual(len(self.events), 3)  # Rendering did not submit a control.

    def test_replacement_trims_obsolete_future_and_release_hides_arrow(self):
        m, s = self.setup_case()
        m.record_control(76, {'velocity_xz': [-.6, 0]})
        self.assertEqual(m.target_velocity_at(s, 150), (-.6, 0))
        self.assertEqual(m.target_velocity_at(s, 51), (0, .6))
        m.record_control(96, {})
        render(s, 96, np.zeros(3))
        self.assertEqual(self.drawn[-1], {'visible': False})
        render(s, 0, np.zeros(3))
        self.assertTrue(self.drawn[-1]['visible'])

    def test_keyboard_preview_uses_requested_direction_and_replay_keeps_history(self):
        m, s = self.setup_case()
        root = np.array([3., .8, 4.])
        render(s, 0, root, True)
        np.testing.assert_allclose(self.drawn[-1]['root_velocity'], [0, 0, -.6])
        np.testing.assert_allclose(self.drawn[-1]['root_pos'], root)
        render(s, 0, root)
        np.testing.assert_allclose(self.drawn[-1]['root_velocity'], [0, 0, .6])
        self.assertEqual(len(self.events), 3)

    def test_history_copies_values_and_new_history_has_no_stale_direction(self):
        h = ControlHistory()
        v = [0, .6]
        h.replace_tail(0, {'velocity_xz': v})
        v[0] = 9
        self.assertEqual(h.at(0), (0, .6))
        self.assertIsNone(ControlHistory().at(0))

    def test_seek_clears_held_keys_and_next_edit_uses_replayed_direction(self):
        m, s = self.setup_case()
        highlights = []
        s.client = NS(client_id=1, timeline=NS(set_highlighted_arrow_keys=highlights.append),
                      add_notification=lambda **kw: None)
        s.mm2_arrow_keys_pressed = {'ArrowLeft'}
        m._seek_playback_impl(NS(sonic_output=None), s, 0)
        self.assertEqual(s.mm2_arrow_keys_pressed, set())
        self.assertEqual(highlights, [[]])
        camera = Camera()
        camera.client_active = lambda _: True
        camera.client_sessions = {1: s}
        camera.on_arrow_key_press(1, {'ArrowUp'})
        np.testing.assert_allclose(s.gui_elements.gui_target_root_velocity.value, (0, .8))
        self.assertIsNone(s.mm2_keyboard_edit_frame)

    def test_live_preview_survives_ticks_but_seek_restores_recorded_direction(self):
        m, s = self.setup_case()
        m.velocity_preview = (.8, 0.)
        render(s, 0, np.zeros(3))
        np.testing.assert_allclose(self.drawn[-1]['root_velocity'], [.8, 0, 0])
        s.client = NS(client_id=1)
        m._seek_playback_impl(NS(sonic_output=None), s, 0)
        render(s, 0, np.zeros(3))
        np.testing.assert_allclose(self.drawn[-1]['root_velocity'], [0, 0, .6])


if __name__ == '__main__':
    unittest.main()
