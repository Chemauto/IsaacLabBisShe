#include "FSM/State_RLBase.h"
#include "unitree_articulation.h"
#include "isaaclab/envs/mdp/observations/observations.h"
#include "isaaclab/envs/mdp/actions/joint_actions.h"
#include <algorithm>

namespace
{
void apply_runtime_command_ranges(const YAML::Node& state_cfg, YAML::Node* env_cfg)
{
    const auto runtime_ranges = state_cfg["command_ranges"];
    if (!runtime_ranges) {
        return;
    }

    auto base_velocity = (*env_cfg)["commands"]["base_velocity"];
    if (!base_velocity) {
        return;
    }

    for (const auto& key : {"lin_vel_x", "lin_vel_y", "ang_vel_z", "heading"}) {
        if (runtime_ranges[key]) {
            (*env_cfg)["commands"]["base_velocity"]["ranges"][key] = runtime_ranges[key];
        }
    }
}
}

namespace isaaclab{
REGISTER_OBSERVATION(keyboard_velocity_commands){
    const std::string key = FSMState::keyboard->key();
    static auto cfg = env->cfg["commands"]["base_velocity"]["ranges"];
    static const std::vector<float> min_cmd = {
        cfg["lin_vel_x"][0].as<float>(),
        cfg["lin_vel_y"][0].as<float>(),
        cfg["ang_vel_z"][0].as<float>()
    };
    static const std::vector<float> max_cmd = {
        cfg["lin_vel_x"][1].as<float>(),
        cfg["lin_vel_y"][1].as<float>(),
        cfg["ang_vel_z"][1].as<float>()
    };
    static std::vector<float> filtered_cmd = {0.0f, 0.0f, 0.0f};
    constexpr float smoothing = 0.2f;
    std::vector<float> cmd = {0.0f, 0.0f, 0.0f};

    if (key.size() == 1) {
        switch (key[0]) {
            case 'w':
                cmd[0] = max_cmd[0];
                break;
            case 's':
                cmd[0] = min_cmd[0];
                break;
            case 'a':
                cmd[1] = max_cmd[1];
                break;
            case 'd':
                cmd[1] = min_cmd[1];
                break;
            case 'q':
                cmd[2] = max_cmd[2];
                break;
            case 'e':
                cmd[2] = min_cmd[2];
                break;
            default:
                break;
        }
    }

    for (size_t i = 0; i < cmd.size(); ++i) {
        filtered_cmd[i] += smoothing * (cmd[i] - filtered_cmd[i]);
        filtered_cmd[i] = std::clamp(filtered_cmd[i], min_cmd[i], max_cmd[i]);
    }
    return filtered_cmd;
}
}

State_RLBase::State_RLBase(int state_mode, std::string state_string)
: FSMState(state_mode, state_string) 
{
    auto cfg = param::config["FSM"][state_string];
    auto policy_dir = param::parser_policy_dir(cfg["policy_dir"].as<std::string>());
    auto env_cfg = YAML::LoadFile(policy_dir / "params" / "deploy.yaml");
    const auto height_map_topic = cfg["height_map_topic"] ? cfg["height_map_topic"].as<std::string>() : "rt/heightmap";
    apply_runtime_command_ranges(cfg, &env_cfg);

    env = std::make_unique<isaaclab::ManagerBasedRLEnv>(
        env_cfg,
        std::make_shared<unitree::BaseArticulation<LowState_t::SharedPtr>>(
            FSMState::lowstate,
            FSMState::sportstate,
            FSMState::heightmap,
            "rt/sportmodestate",
            height_map_topic)
    );
    env->alg = std::make_unique<isaaclab::OrtRunner>(policy_dir / "exported" / "policy.onnx");

    this->registered_checks.emplace_back(
        std::make_pair(
            [&]()->bool{ return isaaclab::mdp::bad_orientation(env.get(), 1.0); },
            FSMStringMap.right.at("Passive")
        )
    );
}

void State_RLBase::run()
{
    auto action = env->action_manager->processed_actions();
    for(int i(0); i < env->robot->data.joint_ids_map.size(); i++) {
        lowcmd->msg_.motor_cmd()[env->robot->data.joint_ids_map[i]].q() = action[i];
    }
}
