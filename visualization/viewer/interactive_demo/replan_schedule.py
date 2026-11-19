"""Choose an unconsumed cut; leave generation, history and splice validation alone."""
from collections import deque
import math


class LiveReplanSchedule:
    # Production splice still enforces its unchanged 8/50=.16 s guard.
    # Keep .24 s beyond measured work to cover that guard and publication jitter.
    margin_s = .24
    tail_s = .80
    minimum_s = {0: .65, 1: .95}

    def __init__(self, prepare_s=.7):
        # Startup prepare_s includes a following window. Keep its original cold
        # double-window allowance until an actual replacement has been measured.
        self.budgets = {0: .65, 1: max(1.6, float(prepare_s) + .8)}
        self.samples = {0: deque(maxlen=8), 1: deque(maxlen=8)}

    def choose(self, consumed, frontier, *, playing, planning=False, target=None, earliest=None):
        if target is not None or not playing:
            frame = int(target) if target is not None else ((int(consumed)+7)//4)*4
            return frame, int(frame % 52 > 20 and not planning)
        if planning:
            # Preserve the original explicit timeline-planning behavior.
            frame = ((int(consumed)+math.ceil(1.2*25)+3)//4)*4
            if frame % 52 > 20:
                frame = ((int(consumed)+math.ceil(max(1.6,self.budgets[1])*25)+3)//4)*4
            return frame, 0
        # Search aligned cuts instead of first jumping ahead and then enlarging
        # that jump based on a different window phase. Never cross the frontier.
        first = (math.floor(float(consumed)/4)+1)*4
        if earliest is not None:
            first = max(first, math.ceil(float(earliest)/4)*4)
        for frame in range(first, int(frontier), 4):
            prefix = frame % 52
            lookahead = int((52-prefix)/25 < self.tail_s)
            if (frame-float(consumed))/25 >= self.budgets[lookahead]:
                return frame, lookahead
        return int(frontier), 0

    def success(self, lookahead, elapsed):
        self.samples[lookahead].append(max(0., float(elapsed)))
        self.budgets[lookahead] = max(self.minimum_s[lookahead],
                                     max(self.samples[lookahead]) + self.margin_s)

    def failure(self, lookahead, elapsed):
        # A late/unsafe candidate never gains permission to overwrite old frames.
        # Retain conservative timing for the existing bounded retry/fallback.
        elapsed = max(0., float(elapsed))
        self.samples[lookahead].append(elapsed)
        self.budgets[lookahead] = max(self.budgets[lookahead], elapsed + .3)
