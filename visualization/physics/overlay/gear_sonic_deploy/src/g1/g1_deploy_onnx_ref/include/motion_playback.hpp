#pragma once

#include <algorithm>

struct MotionFrameAdvanceResult {
    int frame = 0;
    bool completed = false;
    bool waiting_for_append = false;
};

inline MotionFrameAdvanceResult ResolveMotionFrameAdvance(
    int current_frame,
    int timesteps,
    int observation_window,
    bool stream_complete) {
    if (timesteps <= 0) {
        return {0, false, false};
    }

    const int next_frame = std::max(0, current_frame) + 1;
    if (stream_complete) {
        if (next_frame >= timesteps) {
            return {0, true, false};
        }
        return {next_frame, false, false};
    }

    const int first_unsafe_frame =
        std::max(0, timesteps - std::max(0, observation_window));
    if (next_frame >= first_unsafe_frame) {
        return {std::max(0, current_frame), false, true};
    }
    return {next_frame, false, false};
}
