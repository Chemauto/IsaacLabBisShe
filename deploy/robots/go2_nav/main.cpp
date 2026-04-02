#include "FSM/CtrlFSM.h"
#include "FSM/State_FixStand.h"
#include "FSM/State_Passive.h"
#include "State_NavigationRL.h"

std::unique_ptr<LowCmd_t> FSMState::lowcmd = nullptr;
std::shared_ptr<LowState_t> FSMState::lowstate = nullptr;
std::shared_ptr<SportModeState_t> FSMState::sportstate = nullptr;
std::shared_ptr<HeightMap_t> FSMState::heightmap = nullptr;
std::shared_ptr<Keyboard> FSMState::keyboard = std::make_shared<Keyboard>();

void init_fsm_state()
{
    auto lowcmd_sub = std::make_shared<unitree::robot::go2::subscription::LowCmd>();
    usleep(0.2 * 1e6);
    if (!lowcmd_sub->isTimeout()) {
        spdlog::critical("The other process is using the lowcmd channel, please close it first.");
        unitree::robot::go2::shutdown();
    }

    FSMState::lowcmd = std::make_unique<LowCmd_t>();
    FSMState::lowstate = std::make_shared<LowState_t>();
    FSMState::sportstate = std::make_shared<SportModeState_t>();

    std::string height_map_topic = "rt/heightmap";
    const auto navigation_cfg = param::config["FSM"]["Navigation"];
    if (navigation_cfg && navigation_cfg["height_map_topic"]) {
        height_map_topic = navigation_cfg["height_map_topic"].as<std::string>();
    }
    FSMState::heightmap = std::make_shared<HeightMap_t>(height_map_topic);
    FSMState::sportstate->set_timeout_ms(200);
    FSMState::heightmap->set_timeout_ms(200);

    spdlog::info("Waiting for connection to robot...");
    FSMState::lowstate->wait_for_connection();
    spdlog::info("Connected to robot.");
    spdlog::info("Height-map topic: {}", height_map_topic);
}

int main(int argc, char** argv)
{
    auto vm = param::helper(argc, argv);

    std::cout << " --- Unitree Robotics --- \n";
    std::cout << "     Go2 Navigation Controller \n";

    unitree::robot::ChannelFactory::Instance()->Init(0, vm["network"].as<std::string>());

    init_fsm_state();

    auto fsm = std::make_unique<CtrlFSM>(param::config["FSM"]);
    fsm->start();

    std::cout << "Press [L2 + A] to enter FixStand mode.\n";
    std::cout << "And then press [Start] to start navigation control.\n";

    while (true) {
        sleep(1);
    }

    return 0;
}
