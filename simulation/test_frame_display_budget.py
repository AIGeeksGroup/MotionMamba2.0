"""Playback labels may be coalesced; seek, pause and model clocks may not."""
import ast
from pathlib import Path
from types import SimpleNamespace as NS

source = Path(__file__).resolve().parents[1]/'ardy/scripts/interactive_demo/playback.py'
tree = ast.parse(source.read_text())
fn = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)
          and n.name == '_frame_display_due')
fn.decorator_list = []
namespace = {}
exec(compile(ast.fix_missing_locations(ast.Module(body=[fn], type_ignores=[])),
             str(source), 'exec'), namespace)
due = namespace['_frame_display_due']


def test_passive_labels_have_bounded_updates_at_50_hz():
    session = NS(model=NS(is_remote_mm2=True), playing=True, mm2_rendering=True)
    updates = [due(session, i*.02) for i in range(100)]
    assert 17 <= sum(updates) <= 20
    assert updates[0]


def test_seek_preview_and_pause_publish_immediately():
    for playing, rendering in ((True, False), (False, True), (False, False)):
        session = NS(model=NS(is_remote_mm2=True), playing=playing, mm2_rendering=rendering)
        assert due(session, 10)
        assert due(session, 10.001)


def test_original_ardy_display_is_unchanged():
    session = NS(model=NS(is_remote_mm2=False), playing=True, mm2_rendering=True)
    assert due(session, 10)
    assert due(session, 10.001)
