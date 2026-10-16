#include <gtest/gtest.h>

#include "../include/state_logger.hpp"

#include <array>
#include <cstdlib>
#include <optional>
#include <span>
#include <string>
#include <vector>

namespace {

class ScopedHistoryMode {
 public:
    explicit ScopedHistoryMode(const char* mode) {
        const char* current = std::getenv("SONIC_HISTORY_WARMUP");
        if (current != nullptr) old_value_ = current;
        setenv("SONIC_HISTORY_WARMUP", mode, 1);
    }

    ~ScopedHistoryMode() {
        if (old_value_.has_value()) {
            setenv("SONIC_HISTORY_WARMUP", old_value_->c_str(), 1);
        } else {
            unsetenv("SONIC_HISTORY_WARMUP");
        }
    }

 private:
    std::optional<std::string> old_value_;
};

void LogState(StateLogger& logger, double value) {
    const std::array<double, 4> base_quat{1.0, value, 0.0, 0.0};
    const std::array<double, 3> vector3{value, value + 0.1, value + 0.2};
    std::vector<double> body_q{value, value + 1.0};
    std::vector<double> body_dq{value + 2.0, value + 3.0};
    std::vector<double> action{value + 4.0, value + 5.0};
    std::vector<double> temperature(4, value);
    std::vector<double> motor_values(2, value);
    std::vector<double> hand_values(7, value);

    logger.LogFullState(
        base_quat,
        vector3,
        vector3,
        base_quat,
        vector3,
        vector3,
        std::span(body_q),
        std::span(body_dq),
        std::span(action),
        std::span(temperature),
        std::span(motor_values),
        std::span(motor_values),
        std::span(hand_values),
        std::span(hand_values),
        std::span(hand_values),
        std::span(hand_values),
        std::span(hand_values),
        std::span(hand_values));
}

}  // namespace

TEST(StateLoggerHistory, OfficialZeroWarmupPadsFirstObservationWithZeros) {
    ScopedHistoryMode mode("zero");
    StateLogger logger("", 32, 2, 2, 0.02, false);
    LogState(logger, 7.0);

    const auto history = logger.GetLatest(10, 0.02, false);

    ASSERT_EQ(history.size(), 10U);
    for (size_t i = 0; i < history.size() - 1; ++i) {
        EXPECT_EQ(history[i].base_quat, (std::array<double, 4>{0.0, 0.0, 0.0, 0.0}));
        EXPECT_DOUBLE_EQ(history[i].body_q[0], 0.0);
    }
    EXPECT_EQ(history.back().base_quat, (std::array<double, 4>{1.0, 7.0, 0.0, 0.0}));
    EXPECT_DOUBLE_EQ(history.back().body_q[0], 7.0);
    EXPECT_DOUBLE_EQ(history.back().last_action[0], 11.0);
}

TEST(StateLoggerHistory, OfficialZeroWarmupPreservesHistoryOrder) {
    ScopedHistoryMode mode("zero");
    StateLogger logger("", 32, 2, 2, 0.02, false);
    LogState(logger, 1.0);
    LogState(logger, 2.0);
    LogState(logger, 3.0);

    const auto oldest_first = logger.GetLatest(5, false);
    const auto newest_first = logger.GetLatest(5, true);

    ASSERT_EQ(oldest_first.size(), 5U);
    ASSERT_EQ(newest_first.size(), 5U);
    const std::array<double, 5> expected_oldest{0.0, 0.0, 1.0, 2.0, 3.0};
    const std::array<double, 5> expected_newest{3.0, 2.0, 1.0, 0.0, 0.0};
    for (size_t i = 0; i < expected_oldest.size(); ++i) {
        EXPECT_DOUBLE_EQ(oldest_first[i].body_q[0], expected_oldest[i]);
        EXPECT_DOUBLE_EQ(newest_first[i].body_q[0], expected_newest[i]);
    }
}

TEST(StateLoggerHistory, OfficialZeroWarmupWorksWithStridedHistory) {
    ScopedHistoryMode mode("zero");
    StateLogger logger("", 32, 2, 2, 0.02, false);
    for (int value = 1; value <= 5; ++value) {
        LogState(logger, static_cast<double>(value));
    }

    const auto history = logger.GetLatest(4, 0.04, false);

    ASSERT_EQ(history.size(), 4U);
    const std::array<double, 4> expected{0.0, 1.0, 3.0, 5.0};
    for (size_t i = 0; i < expected.size(); ++i) {
        EXPECT_DOUBLE_EQ(history[i].body_q[0], expected[i]);
    }
}

TEST(StateLoggerHistory, RepeatFirstWarmupIsAvailableForIsaacLabABTest) {
    ScopedHistoryMode mode("repeat_first");
    StateLogger logger("", 32, 2, 2, 0.02, false);
    LogState(logger, 7.0);

    const auto history = logger.GetLatest(10, 0.02, false);

    ASSERT_EQ(history.size(), 10U);
    for (const auto& entry : history) {
        EXPECT_DOUBLE_EQ(entry.body_q[0], 7.0);
    }
}

TEST(StateLoggerHistory, InvalidWarmupModeFailsLoudly) {
    ScopedHistoryMode mode("not-a-mode");
    EXPECT_THROW(
        (StateLogger("", 32, 2, 2, 0.02, false)),
        std::invalid_argument);
}
