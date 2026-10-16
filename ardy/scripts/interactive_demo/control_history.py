"""Accepted target velocities on the source timeline, separate from live input."""

from bisect import bisect_right


class ControlHistory:
    def __init__(self):
        # Replace the tuple atomically so rendering never reads a half-written tail.
        self._entries = ()

    def replace_tail(self, frame, control):
        velocity = control.get('velocity_xz')
        velocity = None if velocity is None else tuple(float(v) for v in velocity)
        frame = int(frame)
        prefix = tuple(entry for entry in self._entries if entry[0] < frame)
        self._entries = prefix + ((frame, velocity),)

    def at(self, frame):
        entries = self._entries
        index = bisect_right(entries, int(frame), key=lambda entry: entry[0]) - 1
        return entries[index][1] if index >= 0 else None
