#pragma once

#include "unitree/dds_wrapper/common/Subscription.h"
#include "unitree/dds_wrapper/robots/go2/go2.h"
#include <unitree/idl/go2/HeightMap_.hpp>

using LowCmd_t = unitree::robot::go2::publisher::LowCmd;
using LowState_t = unitree::robot::go2::subscription::LowState;
using SportModeState_t = unitree::robot::go2::subscription::SportModeState;
using HeightMap_t = unitree::robot::SubscriptionBase<unitree_go::msg::dds_::HeightMap_>;
