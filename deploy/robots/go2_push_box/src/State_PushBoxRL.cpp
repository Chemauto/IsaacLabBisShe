#include "State_PushBoxRL.h"

#include <algorithm>
#include <chrono>
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

Eigen::Vector3f projected_gravity_from_lowstate(const LowState_t& lowstate)
{
    const auto& imu = lowstate.msg_.imu_state();
    const Eigen::Quaternionf root_quat_w(
        imu.quaternion()[0],
        imu.quaternion()[1],
        imu.quaternion()[2],
        imu.quaternion()[3]
    );
    return root_quat_w.conjugate() * Eigen::Vector3f(0.0f, 0.0f, -1.0f);
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

State_PushBoxRL::State_PushBoxRL(int state_mode, std::string state_string)
: FSMState(state_mode, state_string),
  cfg_(param::config["FSM"][state_string]),
  push_box_obs_logger_(cfg_["push_box_obs_topic"] ? cfg_["push_box_obs_topic"].as<std::string>() : "rt/push_box_obs")
{
    step_dt_ = cfg_["step_dt"].as<float>(0.02f);
    high_level_decimation_ = cfg_["high_level_decimation"].as<int>(10);

    joint_ids_map_ = read_vector<int>(cfg_, "joint_ids_map");
    default_joint_pos_ = read_vector<float>(cfg_, "default_joint_pos");
    joint_scale_ = read_vector<float>(cfg_, "joint_scale");
    joint_stiffness_ = read_vector<float>(cfg_, "stiffness");
    joint_damping_ = read_vector<float>(cfg_, "damping");
    push_action_clip_ = cfg_["push_action_clip"].as<std::vector<std::vector<float>>>();

    if (joint_ids_map_.size() != kLowLevelActionDim) {
        throw std::runtime_error("joint_ids_map size must match low-level action dim.");
    }
    if (default_joint_pos_.size() != kLowLevelActionDim) {
        throw std::runtime_error("default_joint_pos size must match low-level action dim.");
    }
    if (joint_scale_.size() != kLowLevelActionDim) {
        throw std::runtime_error("joint_scale size must match low-level action dim.");
    }
    if (push_action_clip_.size() != 3) {
        throw std::runtime_error("push action config must be 3-dimensional.");
    }

    const auto high_level_policy_path = resolve_path(cfg_["high_level_policy"].as<std::string>());
    const auto low_level_policy_path = resolve_path(cfg_["low_level_policy"].as<std::string>());
    high_level_policy_ = std::make_unique<isaaclab::OrtRunner>(high_level_policy_path.string());
    low_level_policy_ = std::make_unique<isaaclab::OrtRunner>(low_level_policy_path.string());

    const auto push_box_obs_topic = cfg_["push_box_obs_topic"].as<std::string>("rt/push_box_obs");
    push_box_obs_ = std::make_shared<HeightMap_t>(push_box_obs_topic);
    push_box_obs_->set_timeout_ms(200);

    current_push_actions_.assign(3, 0.0f);
    last_push_actions_.assign(3, 0.0f);
    last_low_level_actions_.assign(kLowLevelActionDim, 0.0f);
    processed_joint_targets_ = default_joint_pos_;
}

std::filesystem::path State_PushBoxRL::resolve_path(const std::string& path) const
{
    std::filesystem::path resolved(path);
    if (resolved.is_relative()) {
        resolved = param::proj_dir / resolved;
    }
    return resolved;
}

void State_PushBoxRL::enter()
{
    for (size_t i = 0; i < joint_stiffness_.size(); ++i) {
        lowcmd->msg_.motor_cmd()[i].kp() = joint_stiffness_[i];
        lowcmd->msg_.motor_cmd()[i].kd() = joint_damping_[i];
        lowcmd->msg_.motor_cmd()[i].dq() = 0.0f;
        lowcmd->msg_.motor_cmd()[i].tau() = 0.0f;
    }

    {
        std::lock_guard<std::mutex> lock(action_mutex_);
        std::fill(current_push_actions_.begin(), current_push_actions_.end(), 0.0f);
        std::fill(last_push_actions_.begin(), last_push_actions_.end(), 0.0f);
        std::fill(last_low_level_actions_.begin(), last_low_level_actions_.end(), 0.0f);
        processed_joint_targets_ = default_joint_pos_;
        high_level_counter_ = 0;
    }

    policy_thread_running_ = true;
    policy_thread_ = std::thread(&State_PushBoxRL::policy_loop, this);
}

void State_PushBoxRL::run()
{
    std::lock_guard<std::mutex> lock(action_mutex_);
    for (size_t i = 0; i < processed_joint_targets_.size(); ++i) {
        const int joint_id = joint_ids_map_[i];
        lowcmd->msg_.motor_cmd()[joint_id].q() = processed_joint_targets_[i];
    }
}

void State_PushBoxRL::exit()
{
    policy_thread_running_ = false;
    if (policy_thread_.joinable()) {
        policy_thread_.join();
    }
}

void State_PushBoxRL::policy_loop()
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

void State_PushBoxRL::update_policy_step()
{
    if (high_level_counter_ == 0) {
        bool has_external_obs = false;
        const auto high_level_obs = build_high_level_obs(&has_external_obs);
        ensure_dim(high_level_obs, kPushObsDim + 3, "push high-level observation");
        if (has_external_obs) {
            const auto raw_push_actions = high_level_policy_->act({{"obs", high_level_obs}});
            process_push_actions(raw_push_actions);
        } else {
            std::fill(current_push_actions_.begin(), current_push_actions_.end(), 0.0f);
            std::fill(last_push_actions_.begin(), last_push_actions_.end(), 0.0f);
        }
    }

    const auto low_level_obs = build_low_level_obs();
    ensure_dim(low_level_obs, kLowLevelObsDim, "push low-level observation");
    const auto raw_low_level_actions = low_level_policy_->act({{"obs", low_level_obs}});
    process_low_level_actions(raw_low_level_actions);

    high_level_counter_ = (high_level_counter_ + 1) % high_level_decimation_;
}

std::vector<float> State_PushBoxRL::build_high_level_obs(bool* has_external_obs)
{
    std::vector<float> obs(kPushObsDim, 0.0f);

    push_box_obs_logger_.poll(push_box_obs_);
    bool push_obs_valid = false;
    if (push_box_obs_ && !push_box_obs_->isTimeout()) {
        std::lock_guard<std::mutex> lock(push_box_obs_->mutex_);
        const auto& data = push_box_obs_->msg_.data();
        if (data.size() == kPushObsDim) {
            obs.assign(data.begin(), data.end());
            push_obs_valid = true;
        }
    }

    if (!push_obs_valid && !warned_missing_push_obs_) {
        spdlog::warn("Push-box observation topic is unavailable or has wrong dim; using zero fallback.");
        warned_missing_push_obs_ = true;
    }
    if (push_obs_valid) {
        warned_missing_push_obs_ = false;
    }

    obs.insert(obs.end(), last_push_actions_.begin(), last_push_actions_.end());
    if (has_external_obs != nullptr) {
        *has_external_obs = push_obs_valid;
    }
    return obs;
}

std::vector<float> State_PushBoxRL::build_low_level_obs()
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

        obs.insert(obs.end(), current_push_actions_.begin(), current_push_actions_.end());

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
            obs.insert(obs.end(), data.begin(), data.end());
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

    return obs;
}

void State_PushBoxRL::process_push_actions(const std::vector<float>& raw_actions)
{
    ensure_dim(raw_actions, 3, "push high-level action");
    for (size_t i = 0; i < raw_actions.size(); ++i) {
        float value = std::clamp(raw_actions[i], push_action_clip_[i][0], push_action_clip_[i][1]);
        current_push_actions_[i] = value;
        last_push_actions_[i] = value;
    }
}

void State_PushBoxRL::process_low_level_actions(const std::vector<float>& raw_actions)
{
    ensure_dim(raw_actions, kLowLevelActionDim, "push low-level action");
    last_low_level_actions_ = raw_actions;

    std::vector<float> processed_actions(kLowLevelActionDim, 0.0f);
    for (size_t i = 0; i < raw_actions.size(); ++i) {
        processed_actions[i] = raw_actions[i] * joint_scale_[i] + default_joint_pos_[i];
    }

    std::lock_guard<std::mutex> lock(action_mutex_);
    processed_joint_targets_ = std::move(processed_actions);
}
