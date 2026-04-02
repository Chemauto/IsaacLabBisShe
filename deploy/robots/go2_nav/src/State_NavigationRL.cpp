#include "State_NavigationRL.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <stdexcept>

#include <eigen3/Eigen/Dense>
#include <spdlog/spdlog.h>

#include "param.h"

namespace
{
template <typename T>
std::vector<T> read_vector(const YAML::Node& node, const char* key)
{
    if (!node[key]) {
        throw std::runtime_error(std::string("Missing config key: ") + key);
    }
    return node[key].as<std::vector<T>>();
}

Eigen::Quaternionf root_quat_from_lowstate(const LowState_t& lowstate)
{
    const auto& imu = lowstate.msg_.imu_state();
    return Eigen::Quaternionf(
        imu.quaternion()[0],
        imu.quaternion()[1],
        imu.quaternion()[2],
        imu.quaternion()[3]
    );
}

Eigen::Vector3f projected_gravity_from_lowstate(const LowState_t& lowstate)
{
    return root_quat_from_lowstate(lowstate).conjugate() * Eigen::Vector3f(0.0f, 0.0f, -1.0f);
}

float yaw_from_quat(const Eigen::Quaternionf& quat)
{
    const float siny_cosp = 2.0f * (quat.w() * quat.z() + quat.x() * quat.y());
    const float cosy_cosp = 1.0f - 2.0f * (quat.y() * quat.y() + quat.z() * quat.z());
    return std::atan2(siny_cosp, cosy_cosp);
}

float wrap_to_pi(float angle)
{
    constexpr float kPi = 3.14159265358979323846f;
    while (angle > kPi) {
        angle -= 2.0f * kPi;
    }
    while (angle < -kPi) {
        angle += 2.0f * kPi;
    }
    return angle;
}

void ensure_dim(const std::vector<float>& values, size_t expected_dim, const std::string& name)
{
    if (values.size() != expected_dim) {
        throw std::runtime_error(
            name + " dim mismatch. Expected " + std::to_string(expected_dim) +
            ", got " + std::to_string(values.size()) + "."
        );
    }
}
}

State_NavigationRL::State_NavigationRL(int state_mode, std::string state_string)
: FSMState(state_mode, state_string),
  cfg_(param::config["FSM"][state_string]),
  goal_command_logger_(cfg_["goal_command_topic"] ? cfg_["goal_command_topic"].as<std::string>() : "rt/navigation_goal")
{
    step_dt_ = cfg_["step_dt"].as<float>(0.02f);
    high_level_decimation_ = cfg_["high_level_decimation"].as<int>(10);
    use_current_height_for_goal_ = cfg_["use_current_height_for_goal"].as<bool>(true);
    latch_last_goal_on_timeout_ = cfg_["latch_last_goal_on_timeout"].as<bool>(true);
    enable_navigation_success_stop_ = cfg_["enable_navigation_success_stop"].as<bool>(false);
    navigation_success_distance_threshold_ = cfg_["navigation_success_distance_threshold"].as<float>(0.20f);
    navigation_success_yaw_threshold_ = cfg_["navigation_success_yaw_threshold"].as<float>(0.15f);
    navigation_success_settle_steps_ = cfg_["navigation_success_settle_steps"].as<int>(3);
    goal_command_timeout_ms_ = cfg_["goal_command_timeout_ms"].as<int>(200);

    const auto default_goal_world = read_vector<float>(cfg_, "default_goal_world");
    if (default_goal_world.size() != kGoalCommandDim) {
        throw std::runtime_error("default_goal_world must be 4-dimensional.");
    }
    std::copy(default_goal_world.begin(), default_goal_world.end(), default_goal_world_.begin());
    current_goal_world_ = default_goal_world_;

    joint_ids_map_ = read_vector<int>(cfg_, "joint_ids_map");
    default_joint_pos_ = read_vector<float>(cfg_, "default_joint_pos");
    joint_scale_ = read_vector<float>(cfg_, "joint_scale");
    joint_stiffness_ = read_vector<float>(cfg_, "stiffness");
    joint_damping_ = read_vector<float>(cfg_, "damping");
    navigation_action_clip_ = cfg_["navigation_action_clip"].as<std::vector<std::vector<float>>>();

    if (joint_ids_map_.size() != kLowLevelActionDim) {
        throw std::runtime_error("joint_ids_map size must match low-level action dim.");
    }
    if (default_joint_pos_.size() != kLowLevelActionDim) {
        throw std::runtime_error("default_joint_pos size must match low-level action dim.");
    }
    if (joint_scale_.size() != kLowLevelActionDim) {
        throw std::runtime_error("joint_scale size must match low-level action dim.");
    }
    if (navigation_action_clip_.size() != 3) {
        throw std::runtime_error("navigation action clip config must be 3-dimensional.");
    }

    const auto high_level_policy_path = resolve_path(cfg_["high_level_policy"].as<std::string>());
    const auto low_level_policy_path = resolve_path(cfg_["low_level_policy"].as<std::string>());
    high_level_policy_ = std::make_unique<isaaclab::OrtRunner>(high_level_policy_path.string());
    low_level_policy_ = std::make_unique<isaaclab::OrtRunner>(low_level_policy_path.string());

    const auto goal_command_topic = cfg_["goal_command_topic"].as<std::string>("rt/navigation_goal");
    goal_command_ = std::make_shared<HeightMap_t>(goal_command_topic);
    goal_command_->set_timeout_ms(goal_command_timeout_ms_);

    current_navigation_actions_.assign(3, 0.0f);
    last_low_level_actions_.assign(kLowLevelActionDim, 0.0f);
    processed_joint_targets_ = default_joint_pos_;
}

std::filesystem::path State_NavigationRL::resolve_path(const std::string& path) const
{
    std::filesystem::path resolved(path);
    if (resolved.is_relative()) {
        resolved = param::proj_dir / resolved;
    }
    return resolved;
}

void State_NavigationRL::enter()
{
    for (size_t i = 0; i < joint_stiffness_.size(); ++i) {
        lowcmd->msg_.motor_cmd()[i].kp() = joint_stiffness_[i];
        lowcmd->msg_.motor_cmd()[i].kd() = joint_damping_[i];
        lowcmd->msg_.motor_cmd()[i].dq() = 0.0f;
        lowcmd->msg_.motor_cmd()[i].tau() = 0.0f;
    }

    {
        std::lock_guard<std::mutex> lock(action_mutex_);
        std::fill(current_navigation_actions_.begin(), current_navigation_actions_.end(), 0.0f);
        std::fill(last_low_level_actions_.begin(), last_low_level_actions_.end(), 0.0f);
        processed_joint_targets_ = default_joint_pos_;
        high_level_counter_ = 0;
        navigation_success_settle_counter_ = 0;
        navigation_success_latched_ = false;
    }
    current_goal_world_ = default_goal_world_;

    policy_thread_running_ = true;
    policy_thread_ = std::thread(&State_NavigationRL::policy_loop, this);
}

void State_NavigationRL::run()
{
    std::lock_guard<std::mutex> lock(action_mutex_);
    for (size_t i = 0; i < processed_joint_targets_.size(); ++i) {
        const int joint_id = joint_ids_map_[i];
        lowcmd->msg_.motor_cmd()[joint_id].q() = processed_joint_targets_[i];
    }
}

void State_NavigationRL::exit()
{
    policy_thread_running_ = false;
    if (policy_thread_.joinable()) {
        policy_thread_.join();
    }
}

void State_NavigationRL::policy_loop()
{
    using clock = std::chrono::high_resolution_clock;
    const auto dt = std::chrono::duration_cast<clock::duration>(std::chrono::duration<double>(step_dt_));
    auto sleep_till = clock::now() + dt;

    while (policy_thread_running_) {
        update_policy_step();
        std::this_thread::sleep_until(sleep_till);
        sleep_till += dt;
    }
}

void State_NavigationRL::update_policy_step()
{
    update_navigation_success_status();
    if (navigation_success_latched_) {
        hold_position();
        return;
    }

    if (high_level_counter_ == 0) {
        bool has_required_obs = false;
        const auto high_level_obs = build_high_level_obs(&has_required_obs);
        ensure_dim(high_level_obs, kHighLevelObsDim, "navigation high-level observation");
        if (has_required_obs) {
            const auto raw_navigation_actions = high_level_policy_->act({{"obs", high_level_obs}});
            process_navigation_actions(raw_navigation_actions);
        } else {
            std::fill(current_navigation_actions_.begin(), current_navigation_actions_.end(), 0.0f);
        }
    }

    const auto low_level_obs = build_low_level_obs();
    ensure_dim(low_level_obs, kLowLevelObsDim, "navigation low-level observation");
    const auto raw_low_level_actions = low_level_policy_->act({{"obs", low_level_obs}});
    process_low_level_actions(raw_low_level_actions);

    high_level_counter_ = (high_level_counter_ + 1) % high_level_decimation_;
}

void State_NavigationRL::update_navigation_success_status()
{
    if (!enable_navigation_success_stop_) {
        navigation_success_settle_counter_ = 0;
        return;
    }

    Eigen::Quaternionf root_quat_w = Eigen::Quaternionf::Identity();
    Eigen::Vector3f robot_pos_w = Eigen::Vector3f::Zero();
    bool sportstate_valid = false;
    {
        if (sportstate && !sportstate->isTimeout()) {
            std::lock_guard<std::mutex> lock(sportstate->mutex_);
            for (int i = 0; i < 3; ++i) {
                robot_pos_w[i] = sportstate->msg_.position()[i];
            }
            sportstate_valid = true;
        }
    }
    if (!sportstate_valid) {
        navigation_success_settle_counter_ = 0;
        return;
    }

    {
        std::lock_guard<std::mutex> lock(lowstate->mutex_);
        root_quat_w = root_quat_from_lowstate(*lowstate);
    }

    std::array<float, kGoalCommandDim> goal_world = current_goal_world_;
    if (use_current_height_for_goal_) {
        goal_world[2] = robot_pos_w.z();
    }

    const float robot_yaw = yaw_from_quat(root_quat_w);
    const float dx = goal_world[0] - robot_pos_w.x();
    const float dy = goal_world[1] - robot_pos_w.y();
    const float goal_distance = std::sqrt(dx * dx + dy * dy);
    const float goal_yaw_error = std::abs(wrap_to_pi(goal_world[3] - robot_yaw));
    const bool reached =
        goal_distance < navigation_success_distance_threshold_ &&
        goal_yaw_error < navigation_success_yaw_threshold_;

    navigation_success_settle_counter_ = reached ? navigation_success_settle_counter_ + 1 : 0;
    if (!navigation_success_latched_ && navigation_success_settle_counter_ >= navigation_success_settle_steps_) {
        navigation_success_latched_ = true;
        spdlog::info(
            "Navigation goal reached. Latching stop with distance {:.4f} m and yaw error {:.4f} rad.",
            goal_distance,
            goal_yaw_error
        );
    }
}

void State_NavigationRL::hold_position()
{
    std::fill(current_navigation_actions_.begin(), current_navigation_actions_.end(), 0.0f);
    std::fill(last_low_level_actions_.begin(), last_low_level_actions_.end(), 0.0f);
    high_level_counter_ = 0;

    std::lock_guard<std::mutex> lock(action_mutex_);
    processed_joint_targets_ = default_joint_pos_;
}

std::vector<float> State_NavigationRL::build_high_level_obs(bool* has_required_obs)
{
    std::vector<float> obs;
    obs.reserve(kHighLevelObsDim);

    Eigen::Quaternionf root_quat_w = Eigen::Quaternionf::Identity();
    Eigen::Vector3f projected_gravity = Eigen::Vector3f(0.0f, 0.0f, -1.0f);
    {
        std::lock_guard<std::mutex> lock(lowstate->mutex_);
        root_quat_w = root_quat_from_lowstate(*lowstate);
        projected_gravity = projected_gravity_from_lowstate(*lowstate);
    }

    Eigen::Vector3f base_lin_vel_b = Eigen::Vector3f::Zero();
    Eigen::Vector3f robot_pos_w = Eigen::Vector3f::Zero();
    bool sportstate_valid = false;
    if (sportstate && !sportstate->isTimeout()) {
        std::lock_guard<std::mutex> lock(sportstate->mutex_);
        for (int i = 0; i < 3; ++i) {
            base_lin_vel_b[i] = sportstate->msg_.velocity()[i];
            robot_pos_w[i] = sportstate->msg_.position()[i];
        }
        sportstate_valid = true;
    }

    if (!sportstate_valid && !warned_missing_navigation_state_) {
        spdlog::warn("Sport-state topic is unavailable; navigation high-level observation will use zero fallback.");
        warned_missing_navigation_state_ = true;
    }
    if (sportstate_valid) {
        warned_missing_navigation_state_ = false;
    }

    if (!latch_last_goal_on_timeout_) {
        current_goal_world_ = default_goal_world_;
    }
    goal_command_logger_.poll(goal_command_);
    if (goal_command_ && !goal_command_->isTimeout()) {
        std::lock_guard<std::mutex> lock(goal_command_->mutex_);
        const auto& goal_data = goal_command_->msg_.data();
        if (goal_data.size() == kGoalCommandDim) {
            for (size_t i = 0; i < kGoalCommandDim; ++i) {
                current_goal_world_[i] = goal_data[i];
            }
            warned_invalid_goal_command_ = false;
        } else if (!warned_invalid_goal_command_) {
            spdlog::warn("Navigation goal topic has wrong dim; keeping previous goal.");
            warned_invalid_goal_command_ = true;
        }
    }
    std::array<float, kGoalCommandDim> goal_world = current_goal_world_;
    if (use_current_height_for_goal_ && sportstate_valid) {
        goal_world[2] = robot_pos_w.z();
    }

    std::array<float, kGoalCommandDim> pose_command = {0.0f, 0.0f, 0.0f, 0.0f};
    if (sportstate_valid) {
        const float robot_yaw = yaw_from_quat(root_quat_w);
        const float dx = goal_world[0] - robot_pos_w.x();
        const float dy = goal_world[1] - robot_pos_w.y();
        pose_command[0] = std::cos(robot_yaw) * dx + std::sin(robot_yaw) * dy;
        pose_command[1] = -std::sin(robot_yaw) * dx + std::cos(robot_yaw) * dy;
        pose_command[2] = goal_world[2] - robot_pos_w.z();
        pose_command[3] = wrap_to_pi(goal_world[3] - robot_yaw);
    }

    obs.insert(obs.end(), base_lin_vel_b.data(), base_lin_vel_b.data() + base_lin_vel_b.size());
    obs.insert(obs.end(), projected_gravity.data(), projected_gravity.data() + projected_gravity.size());
    obs.insert(obs.end(), pose_command.begin(), pose_command.end());

    bool height_scan_valid = false;
    if (heightmap && !heightmap->isTimeout()) {
        std::lock_guard<std::mutex> lock(heightmap->mutex_);
        const auto& data = heightmap->msg_.data();
        if (data.size() == kHeightScanDim) {
            for (const float value : data) {
                obs.push_back(std::clamp(value, -1.0f, 1.0f));
            }
            height_scan_valid = true;
        }
    }

    if (!height_scan_valid) {
        if (!warned_missing_height_scan_) {
            spdlog::warn("Height-map topic is unavailable or has wrong dim; using zero fallback.");
            warned_missing_height_scan_ = true;
        }
        obs.insert(obs.end(), kHeightScanDim, 0.0f);
    } else {
        warned_missing_height_scan_ = false;
    }

    if (has_required_obs != nullptr) {
        *has_required_obs = sportstate_valid && height_scan_valid;
    }
    return obs;
}

std::vector<float> State_NavigationRL::build_low_level_obs()
{
    std::vector<float> obs;
    obs.reserve(kLowLevelObsDim);

    {
        std::lock_guard<std::mutex> lock(lowstate->mutex_);
        const auto& imu = lowstate->msg_.imu_state();
        for (int i = 0; i < 3; ++i) {
            obs.push_back(imu.gyroscope()[i]);
        }

        const auto projected_gravity = projected_gravity_from_lowstate(*lowstate);
        obs.insert(obs.end(), projected_gravity.data(), projected_gravity.data() + projected_gravity.size());

        obs.insert(obs.end(), current_navigation_actions_.begin(), current_navigation_actions_.end());

        for (size_t i = 0; i < joint_ids_map_.size(); ++i) {
            const int joint_id = joint_ids_map_[i];
            obs.push_back(lowstate->msg_.motor_state()[joint_id].q() - default_joint_pos_[i]);
        }
        for (size_t i = 0; i < joint_ids_map_.size(); ++i) {
            const int joint_id = joint_ids_map_[i];
            obs.push_back(lowstate->msg_.motor_state()[joint_id].dq());
        }
    }

    obs.insert(obs.end(), last_low_level_actions_.begin(), last_low_level_actions_.end());

    bool height_scan_valid = false;
    if (heightmap && !heightmap->isTimeout()) {
        std::lock_guard<std::mutex> lock(heightmap->mutex_);
        const auto& data = heightmap->msg_.data();
        if (data.size() == kHeightScanDim) {
            for (const float value : data) {
                obs.push_back(std::clamp(value, -1.0f, 1.0f));
            }
            height_scan_valid = true;
        }
    }

    if (!height_scan_valid) {
        if (!warned_missing_height_scan_) {
            spdlog::warn("Height-map topic is unavailable or has wrong dim; using zero fallback.");
            warned_missing_height_scan_ = true;
        }
        obs.insert(obs.end(), kHeightScanDim, 0.0f);
    }

    return obs;
}

void State_NavigationRL::process_navigation_actions(const std::vector<float>& raw_actions)
{
    ensure_dim(raw_actions, 3, "navigation high-level action");
    for (size_t i = 0; i < raw_actions.size(); ++i) {
        current_navigation_actions_[i] = std::clamp(
            raw_actions[i],
            navigation_action_clip_[i][0],
            navigation_action_clip_[i][1]
        );
    }
}

void State_NavigationRL::process_low_level_actions(const std::vector<float>& raw_actions)
{
    ensure_dim(raw_actions, kLowLevelActionDim, "navigation low-level action");
    last_low_level_actions_ = raw_actions;

    std::vector<float> processed_actions(kLowLevelActionDim, 0.0f);
    for (size_t i = 0; i < raw_actions.size(); ++i) {
        processed_actions[i] = raw_actions[i] * joint_scale_[i] + default_joint_pos_[i];
    }

    std::lock_guard<std::mutex> lock(action_mutex_);
    processed_joint_targets_ = std::move(processed_actions);
}
