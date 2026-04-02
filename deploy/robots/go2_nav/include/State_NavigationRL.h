#pragma once

#include <array>
#include <filesystem>
#include <mutex>
#include <thread>
#include <vector>

#include <yaml-cpp/yaml.h>

#include "FSM/FSMState.h"
#include "isaaclab/algorithms/algorithms.h"
#include "optional_topic_logger.h"

class State_NavigationRL : public FSMState
{
public:
    static constexpr size_t kHighLevelObsDim = 197;
    static constexpr size_t kGoalCommandDim = 4;
    static constexpr size_t kLowLevelObsDim = 232;
    static constexpr size_t kHeightScanDim = 187;
    static constexpr size_t kLowLevelActionDim = 12;

    State_NavigationRL(int state_mode, std::string state_string);

    void enter() override;
    void run() override;
    void exit() override;

private:
    void policy_loop();
    void update_policy_step();

    std::filesystem::path resolve_path(const std::string& path) const;
    std::vector<float> build_high_level_obs(bool* has_required_obs);
    std::vector<float> build_low_level_obs();
    void process_navigation_actions(const std::vector<float>& raw_actions);
    void process_low_level_actions(const std::vector<float>& raw_actions);

    YAML::Node cfg_;

    std::shared_ptr<HeightMap_t> goal_command_;
    OptionalTopicReceiptLogger goal_command_logger_;

    std::unique_ptr<isaaclab::OrtRunner> high_level_policy_;
    std::unique_ptr<isaaclab::OrtRunner> low_level_policy_;

    std::thread policy_thread_;
    bool policy_thread_running_ = false;
    std::mutex action_mutex_;

    float step_dt_ = 0.02f;
    int high_level_decimation_ = 10;
    bool use_current_height_for_goal_ = true;
    int goal_command_timeout_ms_ = 200;
    std::array<float, kGoalCommandDim> default_goal_world_ = {4.8f, 0.0f, 0.0f, 0.0f};

    std::vector<int> joint_ids_map_;
    std::vector<float> default_joint_pos_;
    std::vector<float> joint_scale_;
    std::vector<float> joint_stiffness_;
    std::vector<float> joint_damping_;
    std::vector<std::vector<float>> navigation_action_clip_;

    std::vector<float> current_navigation_actions_;
    std::vector<float> last_low_level_actions_;
    std::vector<float> processed_joint_targets_;

    int high_level_counter_ = 0;
    bool warned_invalid_goal_command_ = false;
    bool warned_missing_navigation_state_ = false;
    bool warned_missing_height_scan_ = false;
};

REGISTER_FSM(State_NavigationRL)
