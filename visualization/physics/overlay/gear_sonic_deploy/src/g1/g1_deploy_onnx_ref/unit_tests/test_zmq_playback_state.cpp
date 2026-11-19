#include <gtest/gtest.h>

#include "../include/input_interface/zmq_manager.hpp"
#include "../include/motion_playback.hpp"

TEST(ZMQManagerPlaybackState, ExplicitPauseWinsWhenControlStarts) {
    EXPECT_FALSE(ZMQManager::ResolveStreamStartPlayback(true, false));
}

TEST(ZMQManagerPlaybackState, ExplicitPlayWinsWhenControlStarts) {
    EXPECT_TRUE(ZMQManager::ResolveStreamStartPlayback(true, true));
}

TEST(ZMQManagerPlaybackState, LegacyPublisherStillAutoPlays) {
    EXPECT_TRUE(ZMQManager::ResolveStreamStartPlayback(false, false));
}

TEST(ZMQManagerPlaybackState, CompletedFiniteMotionClearsStalePlayState) {
    EXPECT_TRUE(ZMQManager::FinitePlaybackJustCompleted(
        true, true, false, false, true, 0));
}

TEST(ZMQManagerPlaybackState, FreshPlayCanRestartFiniteMotion) {
    EXPECT_FALSE(ZMQManager::FinitePlaybackJustCompleted(
        true, true, true, false, true, 0));
}

TEST(ZMQManagerPlaybackState, RollingStreamKeepsWaitingForAppend) {
    EXPECT_FALSE(ZMQManager::FinitePlaybackJustCompleted(
        true, true, false, true, false, 1327));
}

TEST(MotionPlayback, CompleteReferenceReachesLastFrameAndCompletes) {
    constexpr int timesteps = 1375;
    constexpr int observation_window = 47;

    const auto before_last = ResolveMotionFrameAdvance(
        timesteps - 2, timesteps, observation_window, true);
    EXPECT_EQ(before_last.frame, timesteps - 1);
    EXPECT_FALSE(before_last.completed);
    EXPECT_FALSE(before_last.waiting_for_append);

    const auto completed = ResolveMotionFrameAdvance(
        timesteps - 1, timesteps, observation_window, true);
    EXPECT_EQ(completed.frame, 0);
    EXPECT_TRUE(completed.completed);
    EXPECT_FALSE(completed.waiting_for_append);
}

TEST(MotionPlayback, RollingReferenceWaitsForFutureObservationWindow) {
    constexpr int timesteps = 1375;
    constexpr int observation_window = 47;
    constexpr int last_safe_frame = timesteps - observation_window - 1;

    const auto waiting = ResolveMotionFrameAdvance(
        last_safe_frame, timesteps, observation_window, false);
    EXPECT_EQ(waiting.frame, last_safe_frame);
    EXPECT_FALSE(waiting.completed);
    EXPECT_TRUE(waiting.waiting_for_append);
}
