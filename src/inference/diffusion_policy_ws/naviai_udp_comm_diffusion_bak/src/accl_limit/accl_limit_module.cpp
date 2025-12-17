#include "accl_limit/accl_limit_module.h"

#include <cassert>
#include <cstddef>
#include <iostream>
#include <stdexcept>

#define MPC_TEST_SINGLE_JOINT 0

Limit::Limit() : min(0.0), max(0.0)
{
    assert(min - (1e-6) <= max);
}

Limit::Limit(double limit) : min(-std::abs(limit)), max(std::abs(limit))
{
    assert(min - (1e-6) <= max);
}

Limit::Limit(double min_, double max_) : min(min_), max(max_)
{
    assert(min - (1e-6) <= max);
}

Limit::Limit(const std::vector<double> &limit) : min(limit[0]), max(limit[1])
{
    assert(min - (1e-6) <= max);
}

Limit Limit::operator*(double scalar) const
{
    return Limit(min * scalar, max * scalar);
}

Limit Limit::operator/(double scalar) const
{
    return Limit(min / scalar, max / scalar);
}

Limit operator*(double scalar, const Limit &limit)
{
    return Limit(limit.min * scalar, limit.max * scalar);
}

Limit &Limit::operator*=(double scalar)
{
    min *= scalar;
    max *= scalar;
    return *this;
}

Limit &Limit::operator/=(double scalar)
{
    min /= scalar;
    max /= scalar;
    return *this;
}

Accl_Limit_Config::Accl_Limit_Config(double dt_, double position_limit, double speed_limit, double acceleration_limit, double jerk_limit,
                                     double snap_limit)
    : dt(dt_), position(-position_limit, position_limit), speed(-speed_limit, speed_limit),
      acceleration(-acceleration_limit, acceleration_limit), jerk(-jerk_limit, jerk_limit), snap(-snap_limit, snap_limit)
{
}

Accl_Limit_Config::Accl_Limit_Config(double dt_, const Limit &position_limit, double speed_limit, double acceleration_limit,
                                     double jerk_limit, double snap_limit)
    : dt(dt_), position(position_limit), speed(-speed_limit, speed_limit), acceleration(-acceleration_limit, acceleration_limit),
      jerk(-jerk_limit, jerk_limit), snap(-snap_limit, snap_limit)
{
}

Accl_Limit_Config::Accl_Limit_Config(double dt_, const Limit &position_limit, const Limit &speed_limit, const Limit &acceleration_limit,
                                     const Limit &jerk_limit, const Limit &snap_limit)
    : dt(dt_), position(position_limit), speed(speed_limit), acceleration(acceleration_limit), jerk(jerk_limit), snap(snap_limit)
{
}

Accl_Limit_Config Accl_Limit_Config::operator*(double scalar) const
{
    return Accl_Limit_Config{dt, position * scalar, speed * scalar, acceleration * scalar, jerk * scalar, snap * scalar};
}

Accl_Limit_Config &Accl_Limit_Config::operator*=(double scalar)
{
    position *= scalar;
    speed *= scalar;
    acceleration *= scalar;
    jerk *= scalar;
    snap *= scalar;
    return *this;
}

State::State(double position_, double speed_, double acceleration_, double jerk_)
    : state(Eigen::Vector4d(position_, speed_, acceleration_, jerk_))
{
}

State::State(const Eigen::Vector4d &initial) : state(initial)
{
}

double State::position(void) const
{
    return state(0);
}

double State::speed(void) const
{
    return state(1);
}

double State::acceleration(void) const
{
    return state(2);
}

double State::jerk(void) const
{
    return state(3);
}

void State::limitBounds(const Eigen::Vector4d &lb, const Eigen::Vector4d &ub)
{
    state = state.cwiseMin(ub).cwiseMax(lb);
}

void State::limitBounds(const State &lb, const State &ub)
{
    limitBounds(lb.state, ub.state);
}

std::ostream &operator<<(std::ostream &os, const State &obj)
{
    os << "State: [position=" << obj.position() << ", speed=" << obj.speed() << ", acceleration=" << obj.acceleration()
       << ", jerk=" << obj.jerk();
    return os;
}

Accl_Limit_Smoother::Accl_Limit_Smoother(const Accl_Limit_Config &config, const std::vector<State> &initialState, bool useMPC,
                                         bool qpVerbose)
    : useMPC_(useMPC), qpVerbose_(qpVerbose), jointNum_(initialState.size()), currentState_(initialState),
      configs_(std::vector<Accl_Limit_Config>(jointNum_, config)), acclLimitOutputUsedCNT_(0)
{
#if not defined USE_OSQP
    if (useMPC)
        throw std::runtime_error("Should define MACRO USE_OSQP when using MPC!");
#endif
    initSmoother_();
}

Accl_Limit_Smoother::Accl_Limit_Smoother(const std::vector<Accl_Limit_Config> &configs, const std::vector<State> &initialState, bool useMPC,
                                         bool qpVerbose)
    : useMPC_(useMPC), qpVerbose_(qpVerbose), jointNum_(configs.size()), currentState_(initialState), configs_(configs),
      acclLimitOutputUsedCNT_(0)
{
    assert(configs.size() == initialState.size());
#if not defined USE_OSQP
    if (useMPC)
        throw std::runtime_error("Should define MACRO USE_OSQP when using MPC!");
#endif
    initSmoother_();
}

const std::vector<Accl_Limit_Config> &Accl_Limit_Smoother::readConfig(void) const
{
    return configs_;
}

void Accl_Limit_Smoother::setConfig(const std::vector<Accl_Limit_Config> &config)
{
    configs_ = config;
    for (size_t i = 0; i < jointNum_; ++i)
        updateConfig_(i);
}

void Accl_Limit_Smoother::setConfig(const Accl_Limit_Config &config, size_t index)
{
    configs_.at(index) = config;
    updateConfig_(index);
}

void Accl_Limit_Smoother::updateConfig_(size_t index)
{
    const Accl_Limit_Config &config = configs_.at(index);
    Eigen::VectorXd ulb(1), uub(1);
    ulb << config.snap.min;
    uub << config.snap.max;
    stateLowerBound_.at(index) = State(config.position.min, config.speed.min, config.acceleration.min, config.jerk.min);
    stateUpperBound_.at(index) = State(config.position.max, config.speed.max, config.acceleration.max, config.jerk.max);
    controlLowerBound_.at(index) = ulb;
    controlUpperBound_.at(index) = uub;
#if defined USE_OSQP
    if (useMPC_)
        mpcSmoothers_.at(index).setBounds(controlLowerBound_.at(index), controlUpperBound_.at(index), stateLowerBound_.at(index).state,
                                          stateUpperBound_.at(index).state);
#endif
    if (!useMPC_)
        acclSmoothers_.at(index).initParams(config.speed.max, config.speed.min, config.acceleration.max);
}

const std::vector<State> &Accl_Limit_Smoother::readCurrentState(void) const
{
    return currentState_;
}

void Accl_Limit_Smoother::setCurrentState(const std::vector<State> &states)
{
    currentState_ = states;
}

void Accl_Limit_Smoother::setCurrentState(const State &state, size_t index)
{
    currentState_.at(index) = state;
}

void Accl_Limit_Smoother::initSmoother_(void)
{
    for (size_t i = 0; i < jointNum_; ++i)
    {
        const Accl_Limit_Config &config = configs_.at(i);
        Eigen::VectorXd ulb(1), uub(1);
        ulb << config.snap.min;
        uub << config.snap.max;
        stateLowerBound_.emplace_back(config.position.min, config.speed.min, config.acceleration.min, config.jerk.min);
        stateUpperBound_.emplace_back(config.position.max, config.speed.max, config.acceleration.max, config.jerk.max);
        controlLowerBound_.emplace_back(ulb);
        controlUpperBound_.emplace_back(uub);
    }
    lastOutput.resize(jointNum_);
    acclLimitOutput_.resize(jointNum_);
    if (useMPC_)
    {
#if defined USE_OSQP
        Eigen::Vector4d Q_diag_elements(10, 1, 0, 0);
        for (size_t i = 0; i < jointNum_; ++i)
        {
            mpcSmoothers_.emplace_back(LinearChainIntegrator(configs_.at(i).dt), 150, 15, Q_diag_elements.asDiagonal(),
                                       0.02 * Eigen::MatrixXd::Identity(1, 1), 1, stateUpperBound_.at(i).state,
                                       stateLowerBound_.at(i).state, controlUpperBound_.at(i), controlLowerBound_.at(i));
            if (auto res = mpcSmoothers_.at(i).initSolver(qpVerbose_); res != LinearMPC_Smoother::SolverState::Success)
            {
                std::cerr << "Get fialed in initializing solver: " << LinearMPC_Smoother::toString(res) << std::endl;
                throw std::runtime_error("Failed in initializing OSQP solver.");
            }
        }
#endif
    }
    else
    {
        for (size_t i = 0; i < jointNum_; ++i)
        {
            const Accl_Limit_Config &config = configs_.at(i);
            assert(std::abs(config.acceleration.max + config.acceleration.min) < 1e-3);
            acclSmoothers_.emplace_back(config.speed.max, config.speed.min, config.acceleration.max);
        }
    }
}

void Accl_Limit_Smoother::rangeStateLimits_(void)
{
    for (size_t i = 0; i < jointNum_; ++i)
        currentState_.at(i).limitBounds(stateLowerBound_.at(i), stateUpperBound_.at(i));
}

static double getMaxAbsValue(const Eigen::VectorXd &x)
{
    return x.lpNorm<Eigen::Infinity>();
}

static double getMaxAbsValue(const Eigen::VectorXd &x, const Eigen::VectorXd &y)
{
    return std::max(getMaxAbsValue(x), getMaxAbsValue(y));
}

static bool hasInfinity(const Eigen::VectorXd &vec)
{
    return (vec.array().isInf()).any();
}

std::vector<Eigen::VectorXd> Accl_Limit_Smoother::smooth(const std::vector<State> &target)
{
    assert(target.size() == jointNum_);
    if (useMPC_)
    {
#if defined USE_OSQP
#if (defined MPC_TEST_SINGLE_JOINT && MPC_TEST_SINGLE_JOINT == 1)
        for (size_t i = 0; i < 1; ++i)
#else
        for (size_t i = 0; i < jointNum_; ++i)
#endif
        {
            Eigen::VectorXd output;
            if (auto state = mpcSmoothers_.at(i).solve(currentState_.at(i).state, target.at(i).state, output);
                state != LinearMPC_Smoother::SolverState::Success)
            {
                std::cerr << "Get invalid solver state in " << i << "th joint: " << LinearMPC_Smoother::toString(state) << std::endl;
                continue;
            }
            else if (output.hasNaN() || hasInfinity(output) ||
                     (getMaxAbsValue(output) > 1.1 * getMaxAbsValue(controlLowerBound_.at(i), controlUpperBound_.at(i))))
            {
                std::cerr << "Target for " << i << "th joint: " << target.at(i).state.transpose() << std::endl;
                std::cerr << "Current state for " << i << "th joint: " << currentState_.at(i).state.transpose() << std::endl;
                std::cerr << "QP upper bound in " << i << "th joint: " << mpcSmoothers_.at(i).getBound(false).transpose() << std::endl;
                std::cerr << "QP lower bound in " << i << "th joint: " << mpcSmoothers_.at(i).getBound(true).transpose() << std::endl;
                // std::cerr << "QP Constraint Matrix:\n" << mpcSmoothers_.at(i).getLinearConstraintMatrix() << std::endl;
                std::cerr << "Get invalid result in " << i << "th joint: " << output.transpose() << std::endl;
                lastOutput.at(i) = mpcSmoothers_.at(i).limitOutput(output);
            }
            else
                lastOutput.at(i) = mpcSmoothers_.at(i).limitOutput(output);
#if (defined MPC_TEST_SINGLE_JOINT && MPC_TEST_SINGLE_JOINT == 1)
            if (i == 0)
            {
                std::cerr << "Get target for 0th joint: " << target.at(i).state.transpose() << std::endl
                          << "Get current state for 0th joint: " << currentState_.at(i).state.transpose() << std::endl
                          << "Solved control vector for 0th joint: " << output.transpose() << std::endl
                          << std::endl
                          << std::endl;
            }
#endif
        }
#if (defined MPC_TEST_SINGLE_JOINT && MPC_TEST_SINGLE_JOINT == 1)
        for (size_t i = 1; i < jointNum_; ++i)
            lastOutput.at(i) = Eigen::VectorXd(lastOutput.at(0).size()).setZero();
#endif
#endif
    }
    else
    {
        for (size_t i = 0; i < jointNum_; ++i)
        {
            double dt = configs_.at(i).dt;
            DynamicsState current = DynamicsState(currentState_.at(i).position(), currentState_.at(i).speed());
            DynamicsState res =
                acclSmoothers_.at(i).calculateAcclClamp(current, DynamicsState(target.at(i).position(), target.at(i).speed()), dt);
            acclLimitOutput_.at(i).position =
                std::clamp(static_cast<double>(res.position), stateLowerBound_.at(i).position(), stateUpperBound_.at(i).position());
            acclLimitOutput_.at(i).velocity =
                std::clamp(static_cast<double>(res.velocity), stateLowerBound_.at(i).speed(), stateUpperBound_.at(i).speed());
            // Get least square accleration as result
            auto delta = acclLimitOutput_.at(i) - (current + DynamicsState(dt * current.velocity, 0));
            Eigen::Vector2d b(0.5 * dt * dt, dt);
            Eigen::VectorXd ans(1);
            ans << (b.transpose() * Eigen::Vector2d(delta.position, delta.velocity) / (b.transpose() * b))(0);
            lastOutput.at(i) = ans.cwiseMin(controlUpperBound_.at(i)).cwiseMax(controlLowerBound_.at(i));
        }
        acclLimitOutputUsedCNT_ = 0;
    }
    return lastOutput;
}

Eigen::MatrixXd Accl_Limit_Smoother::mapSmoothedToMatrix(const std::vector<Eigen::VectorXd> &vec)
{
    if (vec.empty())
        return Eigen::MatrixXd();
    Eigen::Index rows = vec.size();
    Eigen::Index cols = vec[0].size();
    for (Eigen::Index i = 1; i < rows; ++i)
    {
        if (vec[i].size() != cols)
        {
            std::cerr << "[mapSmoothedToMatrix] Mismatched vector sizes at index " << i << ": expected " << cols << ", got "
                      << vec[i].size() << std::endl;
            throw std::runtime_error("Inconsistent vector sizes in mapSmoothedToMatrix");
        }
    }
    Eigen::MatrixXd mat(rows, cols);
    for (Eigen::Index i = 0; i < rows; ++i)
        mat.row(i).noalias() = vec[i].transpose();
    return mat;
}

const std::vector<State> &Accl_Limit_Smoother::step(const std::vector<double> &control, double dt)
{
    assert(control.size() == jointNum_);
    if (useMPC_)
    {
#if defined USE_OSQP
        for (size_t i = 0; i < jointNum_; ++i)
            currentState_.at(i).state = mpcSmoothers_.at(i).stepDynamics(currentState_.at(i).state, control.at(i), dt);
#endif
    }
    else
    {
        for (size_t i = 0; i < jointNum_; ++i)
        {
            if (acclLimitOutputUsedCNT_ == 0)
                currentState_.at(i) = State(acclLimitOutput_.at(i).position, acclLimitOutput_.at(i).velocity);
            else
                currentState_.at(i).state(0) = currentState_.at(i).position() + configs_.at(i).dt * currentState_.at(i).speed();
        }
        acclLimitOutputUsedCNT_++;
    }
    rangeStateLimits_();
    return currentState_;
}

const std::vector<State> &Accl_Limit_Smoother::step(const Eigen::VectorXd &control, double dt)
{
    assert(control.size() == jointNum_);
    if (useMPC_)
    {
#if defined USE_OSQP
        for (size_t i = 0; i < jointNum_; ++i)
            currentState_.at(i).state = mpcSmoothers_.at(i).stepDynamics(currentState_.at(i).state, control(i), dt);
#endif
    }
    else
        return step(std::vector<double>(jointNum_), dt);
    rangeStateLimits_();
    return currentState_;
}

const std::vector<State> &Accl_Limit_Smoother::step(size_t lastOutputIndex, double dt)
{
    if (useMPC_)
    {
#if defined USE_OSQP
        for (size_t i = 0; i < jointNum_; ++i)
            currentState_.at(i).state = mpcSmoothers_.at(i).stepDynamics(currentState_.at(i).state, lastOutput.at(i)(lastOutputIndex), dt);
#endif
    }
    else
        return step(std::vector<double>(jointNum_), dt);
    rangeStateLimits_();
    return currentState_;
}
