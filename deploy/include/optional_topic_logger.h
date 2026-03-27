#pragma once

#include <string>
#include <utility>

#include <spdlog/spdlog.h>

class OptionalTopicReceiptLogger
{
public:
    explicit OptionalTopicReceiptLogger(std::string topic_name)
    : topic_name_(std::move(topic_name))
    {
    }

    template <typename SubscriptionPtr>
    bool poll(const SubscriptionPtr& subscription)
    {
        if (received_once_ || !subscription || subscription->isTimeout()) {
            return false;
        }

        received_once_ = true;
        spdlog::info("Received optional topic {}", topic_name_);
        return true;
    }

private:
    std::string topic_name_;
    bool received_once_ = false;
};
