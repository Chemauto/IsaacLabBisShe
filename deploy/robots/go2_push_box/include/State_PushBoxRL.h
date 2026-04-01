#pragma once

#include <filesystem>
#include <mutex>
#include <thread>
#include <vector>

#include <yaml-cpp/yaml.h>

#include "FSM/FSMState.h"
#include "isaaclab/algorithms/algorithms.h"
#include "optional_topic_logger.h"

class State_PushBoxRL : public FSMState
{
public:
    static constexpr size_t kPushObsDim = 16;
    static constexpr size_t kLowLevelObsDim = 232;
    static constexpr size_t kHeightScanDim = 187;
    static constexpr size_t kLowLevelActionDim = 12;

    State_PushBoxRL(int state_mode, std::string state_string);

    void enter() override;
    void run() override;
    void exit() override;

private:
    void policy_loop();
    void update_policy_step();
    void update_success_status();
    void hold_position();

    std::filesystem::path resolve_path(const std::string& path) const;
    std::vector<float> build_high_level_obs(bool* has_external_obs);
    std::vector<float> build_low_level_obs();
    void process_push_actions(const std::vector<float>& raw_actions);
    void process_low_level_actions(const std::vector<float>& raw_actions);

    YAML::Node cfg_;

    std::shared_ptr<HeightMap_t> push_box_obs_;
    OptionalTopicReceiptLogger push_box_obs_logger_;

    std::unique_ptr<isaaclab::OrtRunner> high_level_policy_;
    std::unique_ptr<isaaclab::OrtRunner> low_level_policy_;

    std::thread policy_thread_;
    bool policy_thread_running_ = false;
    std::mutex action_mutex_;

    float step_dt_ = 0.02f;
    int high_level_decimation_ = 10;
    bool enable_success_stop_ = true;
    float success_distance_threshold_ = 0.12f;
    float success_yaw_threshold_ = 0.15f;
    int success_settle_steps_ = 4;

    std::vector<int> joint_ids_map_;
    std::vector<float> default_joint_pos_;
    std::vector<float> joint_scale_;
    std::vector<float> joint_stiffness_;
    std::vector<float> joint_damping_;
    std::vector<std::vector<float>> push_action_clip_;

    std::vector<float> current_push_actions_;
    std::vector<float> last_push_actions_;
    std::vector<float> last_low_level_actions_;
    std::vector<float> processed_joint_targets_;

    int high_level_counter_ = 0;
    int success_settle_counter_ = 0;
    bool success_latched_ = false;
    bool warned_missing_push_obs_ = false;
    bool warned_missing_height_scan_ = false;
};

REGISTER_FSM(State_PushBoxRL)
