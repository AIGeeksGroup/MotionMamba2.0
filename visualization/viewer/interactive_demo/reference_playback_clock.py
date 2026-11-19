"""Independent reference preview clock; no simulator telemetry dependency."""


class ReferencePlaybackClock:
    def __init__(self):
        self.previous_time = None

    def reset(self):
        self.previous_time = None

    def advance(self, now, frame, last_frame, playing, fps=25):
        dt = 0.0 if self.previous_time is None else max(0.0, now-self.previous_time)
        self.previous_time = now
        if not playing:
            return frame
        # Discard waiting time at the end of the currently generated buffer.
        # A new window resumes at normal speed instead of catching up in a jump.
        return max(0.0, min(float(last_frame), frame + dt*fps))
