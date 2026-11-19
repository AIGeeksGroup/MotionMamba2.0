#include <gtest/gtest.h>

#include "../include/fk.hpp"
#include "../include/motion_data_reader.hpp"

#include <vector>
#include <fstream>
#include <algorithm>
#include <cmath>

namespace {

double QuaternionAngularDistance(
    const MotionSequence::Quaternion& lhs,
    const MotionSequence::Quaternion& rhs) {
    double dot = 0.0;
    double lhs_norm = 0.0;
    double rhs_norm = 0.0;
    for (int i = 0; i < 4; ++i) {
        dot += lhs[i] * rhs[i];
        lhs_norm += lhs[i] * lhs[i];
        rhs_norm += rhs[i] * rhs[i];
    }
    const double normalized_dot = std::clamp(
        std::abs(dot) / std::sqrt(lhs_norm * rhs_norm), 0.0, 1.0);
    return 2.0 * std::acos(normalized_dot);
}

}  // namespace

TEST(FK, TestFKAndGlobalVelocities) {

    // read some exapmle motion data:
    MotionDataReader motion_reader;
    ASSERT_TRUE(motion_reader.ReadFromCSV("reference/example/"));
    ASSERT_FALSE(motion_reader.motions.empty());

    const auto motion_it = std::find_if(
        motion_reader.motions.begin(), motion_reader.motions.end(),
        [](const auto& motion) { return motion->name == "macarena_001__A545_M"; });
    ASSERT_NE(motion_it, motion_reader.motions.end());
    const auto& motion = *motion_it;

    RobotFK fk("g1/g1_29dof.xml");

    auto num_bodies = motion->GetNumBodies();
    auto timesteps = motion->timesteps;

    std::vector<MotionSequence::Point> body_positions_orig(num_bodies * timesteps);
    std::vector<MotionSequence::Quaternion> body_quaternions_orig(num_bodies * timesteps);

    std::vector<MotionSequence::Velocity> body_lin_velocities_orig(num_bodies * timesteps);
    std::vector<MotionSequence::Velocity> body_ang_velocities_orig(num_bodies * timesteps);

    // record original global space data so we can compare it to the computed data:
    const auto &seq = *motion;
    std::copy(seq.BodyPositions(0), seq.BodyPositions(0) + body_positions_orig.size(), body_positions_orig.begin());
    std::copy(seq.BodyQuaternions(0), seq.BodyQuaternions(0) + body_quaternions_orig.size(), body_quaternions_orig.begin());
    std::copy(seq.BodyLinVelocities(0), seq.BodyLinVelocities(0) + body_lin_velocities_orig.size(), body_lin_velocities_orig.begin());
    std::copy(seq.BodyAngVelocities(0), seq.BodyAngVelocities(0) + body_ang_velocities_orig.size(), body_ang_velocities_orig.begin());
    
    // Keep velocity validation independent from FK reconstruction. The source
    // velocities were generated from these stored global poses, while FK uses
    // joint/root values rounded during CSV export.
    MotionSequence velocity_seq = seq;
    velocity_seq.ComputeGlobalVelocities();

    // Compute FK from the stored root pose and joint angles.
    motion->ComputeFK(fk);

    // sanity check - make sure at least some of the computed components are
    // different from the original data:
    bool found_different_pos = false;
    bool found_different_quat = false;
    bool found_different_lin_vel = false;
    bool found_different_ang_vel = false;
    double angular_squared_error = 0.0;
    double angular_max_error = 0.0;
    size_t angular_value_count = 0;
    for(int f = 0; f < timesteps; ++f) {
        for(int b = 0; b < num_bodies; ++b) {
            if(seq.BodyPositions(f)[b] != body_positions_orig[f * num_bodies + b]) {
                found_different_pos = true;
            }
            if(velocity_seq.BodyLinVelocities(f)[b] != body_lin_velocities_orig[f * num_bodies + b]) {
                found_different_lin_vel = true;
            }
            if(velocity_seq.BodyAngVelocities(f)[b] != body_ang_velocities_orig[f * num_bodies + b]) {
                found_different_ang_vel = true;
            }
            if(seq.BodyQuaternions(f)[b] != body_quaternions_orig[f * num_bodies + b]) {
                found_different_quat = true;
            }
        }
    }
    EXPECT_TRUE(found_different_pos);
    EXPECT_TRUE(found_different_quat);
    EXPECT_TRUE(found_different_lin_vel);
    EXPECT_TRUE(found_different_ang_vel);

    // check the results are close to the original data:
    for(int f = 0; f < timesteps; ++f) {
        for(int b = 0; b < num_bodies; ++b) {
            for(int i = 0; i < 3; ++i) {
                EXPECT_NEAR(seq.BodyPositions(f)[b][i], body_positions_orig[f * num_bodies + b][i], 1e-5);
            }
            // Quaternion signs are not unique: q and -q are the same rotation.
            EXPECT_NEAR(
                QuaternionAngularDistance(seq.BodyQuaternions(f)[b],
                                          body_quaternions_orig[f * num_bodies + b]),
                0.0,
                1e-5);
            for(int i = 0; i < 3; ++i) {
                EXPECT_NEAR(velocity_seq.BodyLinVelocities(f)[b][i], body_lin_velocities_orig[f * num_bodies + b][i], 2e-4);
            }

            for(int i = 0; i < 3; ++i) {
                const double error = std::abs(
                    velocity_seq.BodyAngVelocities(f)[b][i]
                    - body_ang_velocities_orig[f * num_bodies + b][i]);
                angular_squared_error += error * error;
                angular_max_error = std::max(angular_max_error, error);
                ++angular_value_count;
            }
        }
    }

    const double angular_rmse = std::sqrt(angular_squared_error / angular_value_count);
    std::cout << "Angular velocity reconstruction: RMSE=" << angular_rmse
              << " rad/s, max=" << angular_max_error << " rad/s" << std::endl;
    // body_quat.csv and body_ang_vel.csv are separately rounded to six
    // decimals, so differentiating the reloaded quaternions at 50 Hz has a
    // small, measurable quantization floor. Guard both aggregate and outlier
    // error so a real convention or filtering regression still fails.
    EXPECT_LT(angular_rmse, 1e-3);
    EXPECT_LT(angular_max_error, 2e-2);

}
