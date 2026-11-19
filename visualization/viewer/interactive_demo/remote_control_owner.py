"""Explicit, serialized handoff of the single simulated-robot controller."""


def release_remote_owners(demo, new_client_id):
    """Called under the loader lock, only after a manual Take Control click."""
    owners = [(cid, session) for cid, session in list(demo.client_sessions.items())
              if cid != new_client_id and getattr(session.model, 'is_remote_mm2', False)]
    for client_id, session in owners:
        model = session.model
        demo.set_playback_state(client_id, False, 'control_transfer')
        model.prepare_requested = False
        session.stop_playback = True
        thread = session.playback_thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=10.0)
            if thread.is_alive():
                raise RuntimeError('Previous controller is still stopping; no new controller was started.')
        # Wait for any accepted generation/replan to leave its critical section.
        with session.replan_lock, model.generation_lock, model.motion_update_lock:
            model.close()
            session.model = None
        if getattr(demo, 'mm2_physical_owner', None) == client_id:
            demo.mm2_physical_owner = None
        session.mm2_control_blocked = True
        gui = session.gui_elements
        gui.gui_load_model_button.label = 'Take Control'
        gui.gui_chosen_model_md.content = (
            '**Control moved to another tab.** Click **Take Control** to use this tab again.'
        )
        session.client.add_notification(
            title='Control moved', body='Another tab took control. This tab has been paused.',
            color='blue', auto_close_seconds=8.0,
        )
        print(f'[MM2 control] released client {client_id} for client {new_client_id}', flush=True)
    if owners and demo.sonic_output is not None:
        demo.sonic_output.source_control = None
