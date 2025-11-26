#include "accl_limit/joint_accl_limit.h"

#include <algorithm>
#include <cmath>
#include <functional>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <vector>

DynamicsState::DynamicsState() : position(0.0), velocity(0.0){};

DynamicsState::DynamicsState(double position, double velocity) : position(position), velocity(velocity)
{
}

DynamicsState::~DynamicsState()
{
}

DynamicsState::DynamicsState(const DynamicsState &other)
{
    position = other.position;
    velocity = other.velocity;
}

DynamicsState DynamicsState::clamp(const DynamicsState &state, double positionLim, double velocityLim)
{
    if ((positionLim < 0) || (velocityLim < 0))
        throw std::invalid_argument("Cannot pass negative parameter.");
    return DynamicsState(std::clamp(state.position, -positionLim, positionLim), std::clamp(state.velocity, -velocityLim, velocityLim));
}

DynamicsState DynamicsState::operator-(void) const
{
    return DynamicsState(-position, -velocity);
}

DynamicsState DynamicsState::operator+(const DynamicsState &other) const
{
    return DynamicsState(position + other.position, velocity + other.velocity);
}

DynamicsState DynamicsState::operator-(const DynamicsState &other) const
{
    return DynamicsState(position - other.position, velocity - other.velocity);
}

DynamicsState DynamicsState::operator*(double scalar) const
{
    return DynamicsState(position * scalar, velocity * scalar);
}

DynamicsState DynamicsState::operator/(double scalar) const
{
    if (scalar == 0)
        throw std::invalid_argument("Cannot divide by zero.");
    return DynamicsState(position / scalar, velocity / scalar);
}

DynamicsState &DynamicsState::operator=(const DynamicsState &state)
{
    position = state.position;
    velocity = state.velocity;
    return *this;
}

bool DynamicsState::containsNaN(void) const
{
    return ((std::isnan(position)) || (std::isnan(velocity)));
}

std::string DynamicsState::toStr(void) const
{
    std::ostringstream oss;
    oss << "Position: " << position << ", Velocity: " << velocity;
    return oss.str();
}

DynamicsProtecter1D::DynamicsProtecter1D() : maxVelocity_(0.0), minVelocity_(0.0), maxAccl_(0.0), maxAcclTell_(0.0)
{
}

DynamicsProtecter1D::DynamicsProtecter1D(double maxVelocity, double minVelocity, double maxAccl)
    : maxVelocity_(maxVelocity), minVelocity_(minVelocity), maxAccl_(std::abs(maxAccl)), maxAcclTell_(0.95f * maxAccl)
{
}

DynamicsProtecter1D::~DynamicsProtecter1D()
{
}

void DynamicsProtecter1D::initParams(double maxVelocity, double minVelocity, double maxAccl)
{
    maxVelocity_ = maxVelocity;
    minVelocity_ = minVelocity;
    maxAccl_ = maxAccl;
    maxAcclTell_ = 0.95f * maxAccl;
}

double DynamicsProtecter1D::clampVelocity_(double velocity) const
{
    return std::clamp(velocity, minVelocity_, maxVelocity_);
}

DynamicsState DynamicsProtecter1D::calculateAcclStateCvt_(double time, double accl, const DynamicsState &startState)
{
    DynamicsState delta_state = startState;
    delta_state.position += startState.velocity * time + 0.5f * accl * std::pow(time, 2);
    delta_state.velocity += accl * time;
    return delta_state;
}

DynamicsState DynamicsProtecter1D::integrateAcclToPosVel_(const std::vector<double> &timeList, const std::vector<double> &acclList,
                                                          const DynamicsState &initState, double targetTime,
                                                          const DynamicsState &overTimeDefault)
{
    DynamicsState state(initState.position, initState.velocity);
    double integrated_time = 0;
    for (size_t i = 0; i < timeList.size(); ++i)
    {
        integrated_time += timeList[i];
        if (targetTime < integrated_time)
        {
            double time = targetTime - (integrated_time - timeList[i]);
            state = calculateAcclStateCvt_(time, acclList[i], state);
            return state;
        }
        state = calculateAcclStateCvt_(timeList[i], acclList[i], state);
    }
    return overTimeDefault;
}

std::pair<double, double> DynamicsProtecter1D::getCvtVelocityLim_(const DynamicsState &cvtFrame) const
{
    return {minVelocity_ - cvtFrame.velocity, maxVelocity_ - cvtFrame.velocity};
}

DynamicsState DynamicsProtecter1D::handleZerosTargetZeroCurrent_(const DynamicsState &now, double minVel, double maxVel, double time)
{
    auto nearZero = [](double x) { return std::abs(x) <= 1e-4f; };
    bool posZero = nearZero(now.position);
    bool velZero = nearZero(now.velocity);
    // Case 1: both zero
    if (posZero && velZero)
    {
        if (now.velocity >= minVel && now.velocity <= maxVel)
            return DynamicsState();
        else if (now.velocity > maxVel)
            return integrateAcclToPosVel_({now.velocity / maxAccl_, 1000}, {-maxAccl_, 0}, now, time, DynamicsState());
    }
    // Case 2: position \approx 0, velocity \neq 0
    if (posZero && !velZero)
    {
        if (now.position == 0.0f)
            return handleZerosTargetPositiveNegativeState_(now, minVel, maxVel, time);
        else if ((std::pow(now.velocity, 2) >= 2 * maxAccl_ * std::abs(now.position)) ||
                 (std::copysign(1.0f, now.position) * std::copysign(1.0f, now.velocity) > 0))
        {
            double a = std::copysign(maxAccl_, now.velocity);
            double t1 = std::abs(now.velocity) / maxAccl_;
            double t2 = 0.0f;
            if (std::pow(now.velocity, 2) >= 2 * maxAcclTell_ * std::abs(now.position))
            {
                if (std::abs(now.position / now.velocity) > 1e-7f)
                    t2 = std::sqrt(std::pow(now.velocity / maxAccl_, 2) / 2 +
                                   std::copysign(now.position / maxAccl_, now.position / now.velocity));
                else
                    t2 = std::sqrt(std::pow(now.velocity / maxAccl_, 2) / 2);
            }
            else
                t2 = std::sqrt(std::pow(now.velocity / maxAccl_, 2) / 2 + std::abs(now.position / maxAccl_));

            return integrateAcclToPosVel_({t1 + t2, t2}, {-a, a}, now, time, DynamicsState());
        }
        else
            return handleZerosTargetPositiveNegativeState_(now, minVel, maxVel, time);
    }
    // Case 3: position \neq 0, velocity \approx 0
    if (!posZero && velZero)
    {
        double vBound = now.position > 0 ? minVel : maxVel;
        double a = std::copysign(maxAccl_, now.position);
        if (maxAcclTell_ * std::abs(now.position) <= std::pow(vBound, 2))
        {
            double t1 = std::sqrt(std::abs(now.position / maxAccl_ / 2));
            return integrateAcclToPosVel_({t1, t1}, {-a, a}, now, time, DynamicsState());
        }
        else
        {
            velocityLimitFlag_ = true;
            double t1 = std::abs(vBound) / maxAccl_;
            double t2 = std::abs(vBound) > 1e-9f ? (now.position - maxAccl_ * std::pow(t1, 2)) / std::abs(vBound) : 1000.0f;
            return integrateAcclToPosVel_({t1, t2, t1}, {-a, 0, a}, now, time, DynamicsState());
        }
    }
    std::cerr << "Get current position: " << now.position << ", current velocity: " << now.velocity << std::endl;
    throw std::runtime_error("Unknown target state for function handleZerosTargetZeroCurrent_ ");
}

DynamicsState DynamicsProtecter1D::handleCurrentStateOverlimit_(const DynamicsState &now, double minVel, double maxVel, double time)
{
    if (now.velocity > maxVel)
    {
        velocityLimitFlag_ = true;
        return -handleCurrentStateOverlimit_(-now, -maxVel, -minVel, time);
    }
    double t1 = (minVel - now.velocity) / maxAccl_;
    if (time <= t1)
        return calculateAcclStateCvt_(time, maxAccl_, now);
    else
        return calculateAcclClampZerosTarget_(calculateAcclStateCvt_(t1, maxAccl_, now), minVel, maxVel, time - t1);
}

DynamicsState DynamicsProtecter1D::handleZerosTargetPositiveCurrent_(const DynamicsState &now, double minVel, double maxVel, double time)
{
    if (now.position < 0)
        return -handleZerosTargetPositiveCurrent_(-now, -maxVel, -minVel, time);
    double a = maxAccl_, x1 = now.position + std::pow(now.velocity, 2) / (2 * a);
    if (std::pow(minVel, 2) >= x1 * maxAcclTell_)
    {
        double t1 = now.velocity / a, t2 = std::sqrt(x1 / a);
        return integrateAcclToPosVel_({t1 + t2, t2}, {-a, a}, now, time, DynamicsState());
    }
    else
    {
        double t1 = now.velocity / a, t2 = std::abs(minVel) / a;
        double t3 = std::abs(minVel) > 1e-9 ? (now.position - std::pow(now.velocity, 2) / (2 * a)) / minVel : 1000.0;
        t3 = t3 > 0 ? t3 : 0.0;
        return integrateAcclToPosVel_({t1 + t2, t3, t2}, {-a, 0, a}, now, time, DynamicsState());
    }
}

DynamicsState DynamicsProtecter1D::handleZerosTargetPositiveNegativeState_(const DynamicsState &now, double minVel, double maxVel,
                                                                           double time)
{
    if (now.position < 0)
        return -handleZerosTargetPositiveNegativeState_(-now, -maxVel, -minVel, time);
    double a = maxAccl_;
    auto x1 = [minVel](double _a) -> double { return std::pow(minVel, 2) / (2 * _a); };
    auto x2 = [minVel, &now, x1](double _a) -> double {
        return x1(_a) + std::abs(std::pow(minVel, 2) - std::pow(now.velocity, 2)) / (2 * _a);
    };
    auto x3 = [&now](double _a) -> double { return std::pow(now.velocity, 2) / (2 * _a); };
    if (now.position <= x3(maxAcclTell_))
    {
        double t1 = std::abs(now.velocity) / a, t2 = std::sqrt(std::abs(now.position - std::pow(now.velocity, 2) / (2 * a)) / a);
        return integrateAcclToPosVel_({t1 + t2, t2}, {a, -a}, now, time, DynamicsState());
    }
    else if ((x3(maxAcclTell_) < now.position) && (now.position <= x2(maxAcclTell_)))
    {
        double t2 = std::sqrt(std::pow(now.velocity / a, 2) / 2 + now.position / a), t1 = now.velocity / a + t2;
        return integrateAcclToPosVel_({t1, t2}, {-a, a}, now, time, DynamicsState());
    }
    else
    {
        double t1 = (now.velocity - minVel) / a, t3 = std::abs(minVel) / a;
        double t2 = std::abs(minVel) > 1e-9 ? (now.position - x2(a)) / std::abs(minVel) : 1000.0;
        return integrateAcclToPosVel_({t1, t2, t3}, {-a, 0, a}, now, time, DynamicsState());
    }
}

DynamicsState DynamicsProtecter1D::calculateAcclClampZerosTarget_(const DynamicsState &now, double minVel, double maxVel, double time)
{
    velocityLimitFlag_ = false;
    if ((std::abs(now.position) < 1e-4) || (std::abs(now.velocity) <= 1e-4))
        return handleZerosTargetZeroCurrent_(now, minVel, maxVel, time);
    else if ((now.velocity < minVel) || (now.velocity > maxVel))
        return handleCurrentStateOverlimit_(now, minVel, maxVel, time);
    else if (now.position * now.velocity > 0)
        return handleZerosTargetPositiveCurrent_(now, minVel, maxVel, time);
    else if (now.position * now.velocity < 0)
        return handleZerosTargetPositiveNegativeState_(now, minVel, maxVel, time);
    throw std::runtime_error("Unkown target state for function calculateAcclClampZerosTarget_ ");
    return now;
}

DynamicsState DynamicsProtecter1D::calculateAcclClampImpl_(const DynamicsState &currentState, DynamicsState &targetState, double time)
{
    if (currentState.containsNaN() || targetState.containsNaN() || std::isnan(time))
    {
        std::cerr << "current state: (position:" << currentState.position << ", velocity:" << currentState.velocity << ")" << std::endl;
        std::cerr << "target state: (position:" << targetState.position << ", velocity:" << targetState.velocity << ")" << std::endl;
        std::cerr << "plan time: " << time << std::endl;
        if (currentState.containsNaN())
            throw std::runtime_error("Get NaN in given current state.");
        if (targetState.containsNaN())
            throw std::runtime_error("Get NaN in given target state.");
        if (std::isnan(time))
            throw std::runtime_error("Get NaN in given time.");
        return DynamicsState();
    }
    targetState.velocity = clampVelocity_(targetState.velocity);
    auto [minVel, maxVel] = getCvtVelocityLim_(targetState);
    DynamicsState newTargetState = calculateAcclClampZerosTarget_(currentState - targetState, minVel, maxVel, time) + targetState;
    newTargetState.position += targetState.velocity * time;
    lastOutput = newTargetState;
    return newTargetState;
}
