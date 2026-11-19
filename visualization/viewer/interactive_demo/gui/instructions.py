# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Interactive-demo GUI: shared markdown copy.

Single source of truth for the welcome modal, the Instructions tab, and the `h`-key keyboard-
shortcuts modal, so the two shortcut listings can't drift apart.
"""

# Rendered standalone by the `h`-key help modal (gui/io.py) and embedded at
# the end of INSTRUCTIONS_TAB_MD below.
KEYBOARD_SHORTCUTS_MD = (
    "| Key | Action |\n"
    "|:---:|--------|\n"
    "| `Space` | Toggle play / pause |\n"
    "| `j` | Previous frame |\n"
    "| `k` | Next frame |\n"
    "| `r` | Reset auto-follow camera to current frame |\n"
    "| `t` | Toggle target velocity (heading + arrow overlay) |\n"
    "| `Arrow keys` | Steer target velocity (when `t` is enabled) |\n"
    "| `p` | Toggle waypoint control mode |\n"
    "| `z` | Sample kinematic constraints from motion file |\n"
    "| `h` | Show / hide this help |\n"
)

# Shown once per browser in the welcome modal on connect (see client.py's
# on_client_connect); dismissal is persisted client-side via `save_choice`.
QUICK_START_MD = """
##### Camera
Left-drag to rotate · right-drag to pan · scroll to zoom.

##### Playback
`Space` plays/pauses · `j` / `k` step one frame back/forward.
Motion is generated ahead of the playhead automatically while **Auto Replan** is on (Playback tab).

##### Text prompts
Open the **Text** tab, type a prompt (or pick one from the *Prompt List*), then press
**Update Text Prompt**. The new prompt takes effect at the current frame — each prompt
shows as a colored segment on the timeline.

##### Steering
`p` toggles **Waypoint Mode** — click the floor to set where the character should go.
`t` toggles **Target Velocity** — steer with the arrow keys (`↑`/`↓` speed, `←`/`→` turn).

##### Constraints
Click a constraint track in the timeline to pin the character's current pose at that
frame; right-click a marker to remove it. The **Generate** tab can also sample
constraints from a reference motion file.

Press `h` anytime to see all keyboard shortcuts — and see the **Instructions** tab for
the full manual.
"""

# Rendered in the Instructions tab (gui/orchestrator.py).
INSTRUCTIONS_TAB_MD = f"""### Keyboard shortcuts\n{KEYBOARD_SHORTCUTS_MD}"""
