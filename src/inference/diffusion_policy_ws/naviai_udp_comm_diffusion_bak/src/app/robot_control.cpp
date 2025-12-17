#include "app/robot_control.h"
#include "app/naviai_app.h"

#include "utils.h"

#include <cmath>
#include <cstddef>
#include <exception>
#include <iostream>
#include <netdb.h>
#include <ros/package.h>
#include <variant>

#define TEST_ARM_SINGLE_JOINT_MPC 0
#define ESTIMATE_VELOCITY_FROM_POSITION 0

ControlTargetManager::ControlTargetManager(const std::vector<Accl_Limit_Config> &configs, const std::vector<State> &initialState,
                                           double planTime, bool useMPC)
    : smoother_(configs, initialState, useMPC, qpVerbose_), planTime_(planTime), size_(configs.size()), lastSmoothTime_(0.0),
      lastSteppedTime_(0.0)
{
    targets_.resize(configs.size());
}

ControlTargetManager::ControlTargetManager(const Accl_Limit_Config &config, size_t dof, double planTime, bool useMPC)
    : smoother_(config, std::vector<State>(dof), useMPC, qpVerbose_), planTime_(planTime), size_(dof), lastSmoothTime_(0.0),
      lastSteppedTime_(0.0)
{
    targets_.resize(dof);
}

ControlTargetManager::~ControlTargetManager()
{
}

size_t ControlTargetManager::getSize(void)
{
    return size_;
}

void ControlTargetManager::setLimits(size_t index, const Accl_Limit_Config &config)
{
    smoother_.setConfig(config, index);
}

void ControlTargetManager::setLimits(const std::vector<Accl_Limit_Config> &configs)
{
    smoother_.setConfig(configs);
}

void ControlTargetManager::updateCurrentAcclLimitState(size_t index, const State &state)
{
    smoother_.setCurrentState(state, index);
}

void ControlTargetManager::updateCurrentAcclLimitState(const std::vector<State> &states)
{
    smoother_.setCurrentState(states);
}

void ControlTargetManager::updateNewTarget(size_t index, const State &target)
{
    targets_.at(index) = target;
}

void ControlTargetManager::updateNewTarget(const std::vector<State> &targets)
{
    targets_ = targets;
}

void ControlTargetManager::updateNewTarget(const double *const positionTarget, const double *const speedTarget,
                                           const double *const accelerationTarget, const double *const jerkTarget)
{
    bool ptrValidFlags[4] = {positionTarget != nullptr, speedTarget != nullptr, accelerationTarget != nullptr, jerkTarget != nullptr};
    for (size_t i = 0; i < size_; ++i)
    {
        targets_.at(i).state(0) = ptrValidFlags[0] ? positionTarget[i] : 0.0;
        targets_.at(i).state(1) = ptrValidFlags[1] ? speedTarget[i] : 0.0;
        targets_.at(i).state(2) = ptrValidFlags[2] ? accelerationTarget[i] : 0.0;
        targets_.at(i).state(3) = ptrValidFlags[3] ? jerkTarget[i] : 0.0;
    }
}

void ControlTargetManager::updatePlanTime(double planTime)
{
    planTime_ = planTime;
}

const std::vector<State> ControlTargetManager::planNewTarget(void)
{
    auto planTarget = smoother_.smooth(targets_);
    lastSteppedTime_ = lastSmoothTime_ = getUnixTimestampInSeconds();
    return smoother_.step(smoother_.mapSmoothedToMatrix(planTarget).leftCols(1), planTime_);
}

const std::vector<State> ControlTargetManager::planNewTarget(const std::vector<State> &targets)
{
    updateNewTarget(targets);
    return planNewTarget();
}

const std::vector<State> ControlTargetManager::planNewTarget(const double *const positionTarget, const double *const speedTarget,
                                                             const double *const accelerationTarget, const double *const jerkTarget)
{
    updateNewTarget(positionTarget, speedTarget, accelerationTarget, jerkTarget);
    return planNewTarget();
}

const std::vector<Eigen::VectorXd> &ControlTargetManager::getLastPlannedControlVector(void) const
{
    return smoother_.lastOutput;
}

double ControlTargetManager::getLastSmoothTime(void) const
{
    return lastSmoothTime_;
}

double ControlTargetManager::getLastStepTime(void) const
{
    return lastSteppedTime_;
}

const std::vector<State> ControlTargetManager::stepPlannedTarget(const std::vector<double> &control, double dt)
{
    lastSteppedTime_ = getUnixTimestampInSeconds();
    return smoother_.step(control, dt);
}

const std::vector<State> ControlTargetManager::stepPlannedTarget(size_t lastOutputIndex, double dt)
{
    lastSteppedTime_ = getUnixTimestampInSeconds();
    return smoother_.step(lastOutputIndex, dt);
}

ControlTargetManagers::ControlTargetManagers(double dt, bool useMPC)
    : neckTargetManager_(dt, NECK_JOINT_NUM, dt, useMPC), armTargetManager_({ControlTargetManager(dt, LEFT_ARM_JOINT_NUM, dt, useMPC),
                                                                             ControlTargetManager(dt, RIGHT_ARM_JOINT_NUM, dt, useMPC)}),
      handTargetManager_(
          {ControlTargetManager(dt, LEFT_HAND_JOINT_NUM, dt, useMPC), ControlTargetManager(dt, RIGHT_HAND_JOINT_NUM, dt, useMPC)}),
      waistTargetManager_(dt, WAIST_JOINT_NUM, dt, useMPC)
{
}

ControlTargetManagers::~ControlTargetManagers()
{
}

void ControlTargetManagers::setLimits(Robot_Control_Type type, size_t index, const Accl_Limit_Config &config)
{
    getManager_(type).setLimits(index, config);
}

void ControlTargetManagers::setLimits(Robot_Control_Type type, const std::vector<Accl_Limit_Config> &configs)
{
    getManager_(type).setLimits(configs);
}

void ControlTargetManagers::updateCurrentAcclLimitState(Robot_Control_Type type, size_t index, const State &state)
{
    getManager_(type).updateCurrentAcclLimitState(index, state);
}

void ControlTargetManagers::updateCurrentAcclLimitState(Robot_Control_Type type, const std::vector<State> &states)
{
    getManager_(type).updateCurrentAcclLimitState(states);
}

const std::vector<State> ControlTargetManagers::planNewTarget(Robot_Control_Type type, const std::vector<State> &targets, double planTime)
{
    getManager_(type).updatePlanTime(planTime);
    return getManager_(type).planNewTarget(targets);
}

const std::vector<State> ControlTargetManagers::planNewTarget(Robot_Control_Type type, const double *const positionTarget,
                                                              const double *const speedTarget, const double *const accelerationTarget,
                                                              const double *const jerkTarget, double planTime)
{
    getManager_(type).updatePlanTime(planTime);
    return getManager_(type).planNewTarget(positionTarget, speedTarget, accelerationTarget, jerkTarget);
}

const std::vector<Eigen::VectorXd> &ControlTargetManagers::getLastPlannedControlVector(Robot_Control_Type type)
{
    return getManager_(type).getLastPlannedControlVector();
}

double ControlTargetManagers::getLastSmoothTime(Robot_Control_Type type)
{
    return getManager_(type).getLastSmoothTime();
}

double ControlTargetManagers::getLastStepTime(Robot_Control_Type type)
{
    return getManager_(type).getLastStepTime();
}

const std::vector<State> ControlTargetManagers::stepPlannedTarget(Robot_Control_Type type, const std::vector<double> &control, double dt)
{
    return getManager_(type).stepPlannedTarget(control, dt);
}

const std::vector<State> ControlTargetManagers::stepPlannedTarget(Robot_Control_Type type, size_t lastOutputIndex, double dt)
{
    return getManager_(type).stepPlannedTarget(lastOutputIndex, dt);
}

ControlTargetManager &ControlTargetManagers::getManager_(Robot_Control_Type type)
{
    switch (type)
    {
    case Robot_Control_Type::Neck_Target:
        return neckTargetManager_;
    case Robot_Control_Type::Left_Arm_Target:
        return armTargetManager_[0];
    case Robot_Control_Type::Right_Arm_Target:
        return armTargetManager_[1];
    case Robot_Control_Type::Left_Hand_Target:
        return handTargetManager_[0];
    case Robot_Control_Type::Right_Hand_Target:
        return handTargetManager_[1];
    case Robot_Control_Type::Waist_Target:
        return waistTargetManager_;
    default:
        throw std::runtime_error("Unknown TargetType.");
    }
}

ControlTargets::ControlTargets(bool useDoubleBuffer, double dt, bool useMPC) : targetBuffer(useDoubleBuffer), targetManager(dt, useMPC)
{
}

ControlTargets::~ControlTargets()
{
}

Robot_Control_Target::Robot_Control_Target(bool useDoubleBuffer, bool handUsePlan, bool useMPC)
    : targets_(useDoubleBuffer, useMPC ? mpcStepTime_ : stepTime_, useMPC), useMPC_(useMPC), jointStatesReceived_(false),
      startRecvTargetsFlag_(true), haveSetTarget_(false), slowMoveJTarget_({Eigen::VectorXd(7), Eigen::VectorXd(7)}), allowMoving_(true),
      handUsePlanning_(handUsePlan), forceReceived_(false), haveCalculatedMPC_(false)
{
    acclLimitParams_.dt = useMPC ? mpcStepTime_ : stepTime_;
    loadConfig_();
    loadJointLimits_();
    lastRecvTargetTime_ = getUnixTimestampInSeconds() - 100.0;
    jointStatesReceivedTime_ = getUnixTimestampInSeconds() - 100.0;
    forceReceivedTime_ = getUnixTimestampInSeconds() - 100.0;
}

Robot_Control_Target::~Robot_Control_Target()
{
}

void Robot_Control_Target::loadConfig_(void)
{
    packagePath_ = ros::package::getPath("naviai_udp_comm");
    if (packagePath_.empty())
    {
        ROS_ERROR("Failed to find package naviai_udp_comm.");
        return;
    }
    std::string config_path = packagePath_ + "/config/robot.yaml";
    auto config = loadRobotConfig(config_path);
    flags_.setControlFlag(Robot_Control_Type::Neck_Target, config["Control_Tasks"]["Control_Neck"].as<bool>());
    flags_.setControlFlag(Robot_Control_Type::Left_Arm_Target, config["Control_Tasks"]["Control_Arm"].as<bool>());
    flags_.setControlFlag(Robot_Control_Type::Right_Arm_Target, config["Control_Tasks"]["Control_Arm"].as<bool>());
    flags_.setControlFlag(Robot_Control_Type::Left_Hand_Target, config["Control_Tasks"]["Control_Hand"].as<bool>());
    flags_.setControlFlag(Robot_Control_Type::Right_Hand_Target, config["Control_Tasks"]["Control_Hand"].as<bool>());
    flags_.setControlFlag(Robot_Control_Type::Waist_Target, config["Control_Tasks"]["Control_Waist"].as<bool>());
    feedbackTasks_[static_cast<size_t>(Feedback_Tasks_e::Test_Send_Image)] = config["Feedback_Tasks"]["Test_Send_Image"].as<bool>();
    feedbackTasks_[static_cast<size_t>(Feedback_Tasks_e::Send_Joint_Feedback)] = config["Feedback_Tasks"]["Send_Joint_Feedback"].as<bool>();
    feedbackTasks_[static_cast<size_t>(Feedback_Tasks_e::Send_Image_Feedback)] = config["Feedback_Tasks"]["Send_Image_Feedback"].as<bool>();
    feedbackTasks_[static_cast<size_t>(Feedback_Tasks_e::Send_Force_Feedback)] = config["Feedback_Tasks"]["Send_Force_Feedback"].as<bool>();
    usePlanning_ = config["Send_Target_Use_Planning"].as<bool>();
    if (useMPC_)
    {
        acclLimitParams_.speed = Limit(config["Accl_Limit_Params"]["MPC"]["MAX_JOINT_SPEED"].as<double>());
        acclLimitParams_.acceleration = Limit(config["Accl_Limit_Params"]["MPC"]["MAX_JOINT_ACCL"].as<double>());
        acclLimitParams_.jerk = Limit(config["Accl_Limit_Params"]["MPC"]["MAX_JOINT_JERK"].as<double>());
        acclLimitParams_.snap = Limit(config["Accl_Limit_Params"]["MPC"]["MAX_JOINT_SNAP"].as<double>());
    }
    else
    {
        acclLimitParams_.speed = Limit(config["Accl_Limit_Params"]["ACCL_LIMIT"]["MAX_JOINT_SPEED"].as<double>());
        acclLimitParams_.acceleration = Limit(config["Accl_Limit_Params"]["ACCL_LIMIT"]["MAX_JOINT_ACCL"].as<double>());
        acclLimitParams_.jerk = Limit();
        acclLimitParams_.snap = Limit();
    }
}

void Robot_Control_Target::loadAndSetJointTargets_(Robot_Control_Type controlType, size_t offset, size_t count,
                                                   YAML::Node &robotJointLimits)
{
    std::vector<double> startupTargets(count * 2);
    for (size_t i = 0; i < count; ++i)
    {
        const std::string &jointName =
            ((controlType == Robot_Control_Type::Left_Hand_Target) || (controlType == Robot_Control_Type::Right_Hand_Target))
                ? handJointNames[offset + i]
                : jointNames[offset + i];
        if (!robotJointLimits[jointName].IsMap())
        {
            ROS_ERROR_STREAM("Invalid format or missing key in YAML: " << jointName);
            continue;
        }
        double upper = DEG2RAD(robotJointLimits[jointName]["Upper"].as<double>()),
               lower = DEG2RAD(robotJointLimits[jointName]["Lower"].as<double>()),
               startup = DEG2RAD(robotJointLimits[jointName]["Startup"].as<double>());
        targets_.targetManager.setLimits(controlType, i,
                                         Accl_Limit_Config(acclLimitParams_.dt, Limit(lower, upper), acclLimitParams_.speed,
                                                           acclLimitParams_.acceleration, acclLimitParams_.jerk, acclLimitParams_.snap));
        startupTargets.at(i) = startup;
        startupTargets.at(i + count) = 0.0;
    }
    targets_.targetBuffer.writeNewData(controlType, startupTargets, true);
}

void Robot_Control_Target::loadJointLimits_(void)
{
    if (packagePath_.empty())
    {
        ROS_ERROR("Failed to find package naviai_udp_comm.");
        return;
    }
    std::string config_path = packagePath_ + "/config/jointLimits.yaml";
    try
    {
        auto jointLimits = YAML::LoadFile(config_path);
#if defined USING_NAVIAI_ROBOT_WA1_0303
        auto robotJointLimits = jointLimits["WA1_0303"];
#elif defined USING_NAVIAI_ROBOT_H1_PRO
        auto robotJointLimits = jointLimits["H1_Pro"];
#elif defined USING_NAVIAI_ROBOT_WA2_A2_LITE
        auto robotJointLimits = jointLimits["WA2_A2_lite"];
#else
#error "Please define robot select macro: USING_NAVIAI_ROBOT_WA1_0303/USING_NAVIAI_ROBOT_H1_PRO/USING_NAVIAI_ROBOT_WA2_A2_LITE"
#endif
        loadAndSetJointTargets_(Robot_Control_Type::Neck_Target, NECK_INDEX[0], NECK_JOINT_NUM, robotJointLimits);
        loadAndSetJointTargets_(Robot_Control_Type::Left_Arm_Target, LEFT_ARM_INDEX[0], LEFT_ARM_JOINT_NUM, robotJointLimits);
        loadAndSetJointTargets_(Robot_Control_Type::Right_Arm_Target, RIGHT_ARM_INDEX[0], RIGHT_ARM_JOINT_NUM, robotJointLimits);
        loadAndSetJointTargets_(Robot_Control_Type::Waist_Target, WAIST_INDEX[0], WAIST_JOINT_NUM, robotJointLimits);
        auto handJointLimits = jointLimits["Hand"];
        loadAndSetJointTargets_(Robot_Control_Type::Left_Hand_Target, 0, LEFT_HAND_JOINT_NUM, handJointLimits);
        loadAndSetJointTargets_(Robot_Control_Type::Right_Hand_Target, 0, RIGHT_HAND_JOINT_NUM, handJointLimits);
    }
    catch (const YAML::Exception &e)
    {
        ROS_ERROR("Failed to parse YAML file: %s", e.what());
    }
}

bool Robot_Control_Target::getFeedbackTasksFlag(Feedback_Tasks_e task) const
{
    return feedbackTasks_[static_cast<size_t>(task)];
}

void Robot_Control_Target::setMovingPermissionFlag(bool permission)
{
    allowMoving_ = permission;
}

void Robot_Control_Target::updateNewJointState(const std::array<double, BODY_JOINT_NUM> &position,
                                               const std::array<double, BODY_JOINT_NUM> &velocity)
{
    try
    {
        bool positionValid = (position.size() == BODY_JOINT_NUM), velocityValid = (velocity.size() == BODY_JOINT_NUM);
        if ((!positionValid) || (!velocityValid))
        {
            ROS_WARN("Invalid joint_states msg, get position of size %ld and velocity of size %ld.", position.size(), velocity.size());
            if ((!positionValid) && (!velocityValid))
                return;
        }
        size_t index = 0, j = 0;
        constexpr size_t switchIndex[4] = {RIGHT_ARM_INDEX[1], LEFT_ARM_INDEX[1], NECK_INDEX[1], WAIST_INDEX[1]};
        constexpr size_t biasIndex[4] = {RIGHT_ARM_INDEX[0], LEFT_ARM_INDEX[0], NECK_INDEX[0], WAIST_INDEX[0]};
        using ArrayVariant = std::variant<std::reference_wrapper<boost::array<double, 2>>, std::reference_wrapper<boost::array<double, 3>>,
                                          std::reference_wrapper<boost::array<double, 4>>, std::reference_wrapper<boost::array<double, 6>>,
                                          std::reference_wrapper<boost::array<double, 7>>, std::reference_wrapper<boost::array<double, 8>>>;
        std::vector<ArrayVariant> positionArrays = {std::ref(newJointState_.rightArmPosition), std::ref(newJointState_.leftArmPosition),
                                                    std::ref(newJointState_.neckPosition), std::ref(newJointState_.waistPosition)};
        std::vector<ArrayVariant> velocityArrays = {std::ref(newJointState_.rightArmVelocity), std::ref(newJointState_.leftArmVelocity),
                                                    std::ref(newJointState_.neckVelocity), std::ref(newJointState_.waistVelocity)};
        auto setVariantElement = [](auto &arr_wrapper, size_t idx, double value) { arr_wrapper.get().at(idx) = value; };
        for (size_t i = 0; i < BODY_JOINT_NUM; ++i)
        {
            if (i <= switchIndex[index])
                j = i - biasIndex[index];
            else
            {
                index++;
                j = 0;
            }
            double tmpPos = positionValid ? static_cast<double>(position[i]) : 0.0,
                   tmpVel = velocityValid ? static_cast<double>(velocity[i]) : 0.0;
            if (!(getNumberIsNormal(tmpPos) && getNumberIsNormal(tmpVel)))
            {
                ROS_ERROR("Get NaN/Inf in converted joint states, get position: %s, velocity: %s", cvtDataString(position).c_str(),
                          cvtDataString(velocity).c_str());
                return;
            }
            if (positionValid)
                std::visit([j, tmpPos, &setVariantElement](auto &arr) { setVariantElement(arr, j, tmpPos); }, positionArrays[index]);
            if (velocityValid)
                std::visit([j, tmpVel, &setVariantElement](auto &arr) { setVariantElement(arr, j, tmpVel); }, velocityArrays[index]);
        }
        if (!jointStatesReceived_)
            setPlanCurrentPointFromJointStates_();
        jointStatesReceived_ = true;
        jointStatesReceivedTime_ = getUnixTimestampInSeconds();
    }
    catch (const std::exception &e)
    {
        ROS_ERROR("Get error %s while updating new joint states.", e.what());
    }
}

template <typename T, typename U,
          typename std::enable_if<std::is_floating_point<T>::value && std::is_floating_point<U>::value, int>::type = 0>
static inline void firstOrderLowPassFilterWriteLocal_(T &lastData, T newData, U newDataRate)
{
    lastData = lastData * static_cast<T>(1 - newDataRate) + newData * static_cast<T>(newDataRate);
}

void Robot_Control_Target::updateNewForceFeedback(const Eigen::Vector3d &force, const Eigen::Vector3d &torque, bool isLeft)
{
    try
    {
        forceReceived_ = true;
        if (!getFeedbackTasksFlag(Feedback_Tasks_e::Send_Force_Feedback))
            return;
        forceReceivedTime_ = getUnixTimestampInSeconds();
        int base = isLeft ? 0 : 6;
        firstOrderLowPassFilterWriteLocal_(forceFeedback_.at(base + 0), static_cast<double>(force.x()), 0.05);
        firstOrderLowPassFilterWriteLocal_(forceFeedback_.at(base + 1), static_cast<double>(force.y()), 0.05);
        firstOrderLowPassFilterWriteLocal_(forceFeedback_.at(base + 2), static_cast<double>(force.z()), 0.05);
        firstOrderLowPassFilterWriteLocal_(forceFeedback_.at(base + 3), static_cast<double>(torque.x()), 0.05);
        firstOrderLowPassFilterWriteLocal_(forceFeedback_.at(base + 4), static_cast<double>(torque.y()), 0.05);
        firstOrderLowPassFilterWriteLocal_(forceFeedback_.at(base + 5), static_cast<double>(torque.z()), 0.05);
    }
    catch (const std::exception &e)
    {
        ROS_ERROR("Get error %s while updating new force feedback.", e.what());
    }
}

void Robot_Control_Target::setPlanCurrentPointFromJointStates_(void)
{
    try
    {
        for (size_t i = 0; i < 7; i++)
        {
            targets_.targetManager.updateCurrentAcclLimitState(
                Robot_Control_Type::Left_Arm_Target, i, State(newJointState_.leftArmPosition.at(i), newJointState_.leftArmVelocity.at(i)));
            targets_.targetManager.updateCurrentAcclLimitState(
                Robot_Control_Type::Right_Arm_Target, i,
                State(newJointState_.rightArmPosition.at(i), newJointState_.rightArmVelocity.at(i)));
        }
        for (size_t i = 0; i < 2; i++)
            targets_.targetManager.updateCurrentAcclLimitState(Robot_Control_Type::Neck_Target, i,
                                                               State(newJointState_.neckPosition.at(i), newJointState_.neckVelocity.at(i)));
        for (size_t i = 0; i < BODY_JOINT_NUM - 14 - 2; i++)
            targets_.targetManager.updateCurrentAcclLimitState(
                Robot_Control_Type::Waist_Target, i, State(newJointState_.waistPosition.at(i), newJointState_.waistVelocity.at(i)));
    }
    catch (const std::exception &e)
    {
        LOG_ERROR() << "Get exception while setting accl limit state: " << e.what();
    }
}

bool Robot_Control_Target::checkArmControlTargetError(bool isLeft)
{
    Eigen::VectorXd feedback(7);
    const auto &position = isLeft ? newJointState_.leftArmPosition : newJointState_.rightArmPosition;
    for (size_t i = 0; i < 7; ++i)
        feedback(i) = position.at(i);
    Eigen::VectorXd &target = slowMoveJTarget_[isLeft ? 0 : 1];
    return (feedback - target).norm() < 0.05;
}

void Robot_Control_Target::setGenericTarget_(Robot_Control_Type controlType, double *const positionTarget, double *const velocityTarget,
                                             size_t dof, bool usePlan, bool needJointState, bool &validFlag,
                                             std::function<void(size_t, double)> optionalCallback)
{
    if (!allowMoving_)
        return;
    auto target = targets_.targetBuffer.readData(controlType);
    bool controlFlag = flags_.getControlFlag(controlType);
    bool ptrValidFlags[3] = {positionTarget != nullptr, velocityTarget != nullptr, optionalCallback != nullptr};
    if (!needJointState || jointStatesReceived_)
    {
        if (usePlan)
        {
            auto res = targets_.targetManager.planNewTarget(controlType, target, target + dof, nullptr, nullptr, planTime_);
            for (size_t i = 0; i < dof; ++i)
            {
                if (ptrValidFlags[0])
                    positionTarget[i] = res.at(i).position();
                if (ptrValidFlags[1])
                    velocityTarget[i] = res.at(i).speed();
            }
        }
        else
        {
#if (defined TEST_ARM_SINGLE_JOINT_MPC) && (TEST_ARM_SINGLE_JOINT_MPC == 1)
            std::cout << "Waiting for MoveJ to move to correct position." << std::endl;
#endif
            for (size_t i = 0; i < dof; ++i)
            {
                if (ptrValidFlags[0])
                    positionTarget[i] = target[i];
                if (ptrValidFlags[1])
                    velocityTarget[i] = 0.0;
            }
        }
        for (size_t i = 0; i < dof; ++i)
            if (ptrValidFlags[2])
                optionalCallback(i, positionTarget[i]);
        validFlag = controlFlag && targets_.targetBuffer.getDataValid(controlType);
        targets_.targetBuffer.switchBuffer(controlType);
    }
    else if (needJointState)
    {
        ROS_WARN("Have not received joint states.");
    }
}

static const std::vector<State> getSteppedTarget_(ControlTargetManagers &manager, Robot_Control_Type type, double mpcStepTime,
                                                  double stepTime, bool haveCalculatedMPC = true)
{
    std::vector<State> res;
    auto myRound = [](double x) -> int { return (x > 0) ? static_cast<int>(x + 0.5) : static_cast<int>(x - 0.5); };
    const double currentTimeStamp = getUnixTimestampInSeconds();
    double mpcSteps = (currentTimeStamp - manager.getLastSmoothTime(type)) / mpcStepTime;
    if (!haveCalculatedMPC)
        return res;
    if (mpcSteps > 2)
    {
#if (defined TEST_ARM_SINGLE_JOINT_MPC) && (TEST_ARM_SINGLE_JOINT_MPC == 0)
        ROS_ERROR("%lf seconds have passed since last MPC calculation.", currentTimeStamp - manager.getLastSmoothTime(type));
        return res;
#endif
    }
    else
    {
        int controlStep = myRound((currentTimeStamp - manager.getLastSmoothTime(type)) / stepTime),
            steppedControlStep = myRound((manager.getLastStepTime(type) - manager.getLastSmoothTime(type)) / stepTime);
        if (controlStep == steppedControlStep)
            return res;
        else if (controlStep < steppedControlStep)
        {
            ROS_ERROR("Get invalid step time: have stepped %d times but need to step %d times", steppedControlStep, controlStep);
            return res;
        }
        double t1 = (controlStep + 1) * stepTime / mpcStepTime, t2 = controlStep * stepTime / mpcStepTime;
        if (std::abs(std::floor(t1) - std::floor(t2)) < 1e-3)
        {
            res = manager.stepPlannedTarget(type, std::floor(mpcSteps), (controlStep - steppedControlStep) * stepTime);
        }
        else
        {
            double dt2 = (t1 - std::floor(t1)) * mpcStepTime, dt1 = stepTime - dt2;
            res = manager.stepPlannedTarget(type, std::floor(mpcSteps), dt1);
            res = manager.stepPlannedTarget(type, std::floor(mpcSteps) + 1, dt2);
        }
    }
    return res;
}

void Robot_Control_Target::stepGenericTarget_(Robot_Control_Type controlType, double *const positionTarget, double *const velocityTarget,
                                              size_t dof, bool usePlan, bool needJointState, bool &validFlag)
{
    if (!allowMoving_)
        return;
    auto target = targets_.targetBuffer.readData(controlType);
    bool controlFlag = flags_.getControlFlag(controlType);
    bool ptrValidFlags[2] = {positionTarget != nullptr, velocityTarget != nullptr};

    if (!needJointState || jointStatesReceived_)
    {
        if (usePlan)
        {
            std::vector<State> res = getSteppedTarget_(targets_.targetManager, controlType, mpcStepTime_, stepTime_);
            if (res.size() < dof)
                return;
            for (size_t i = 0; i < dof; ++i)
            {
                if (ptrValidFlags[0])
                    positionTarget[i] = res.at(i).position();
                if (ptrValidFlags[1])
                    velocityTarget[i] = res.at(i).speed();
            }
        }
    }
}

double *const Robot_Control_Target::getTargetPtr_(Robot_Control_Type type, bool isPosition)
{
    switch (type)
    {
    case Robot_Control_Type::Left_Arm_Target:
        return isPosition ? finalTarget_.leftArmPosition.data() : finalTarget_.leftArmVelocity.data();
    case Robot_Control_Type::Right_Arm_Target:
        return isPosition ? finalTarget_.rightArmPosition.data() : finalTarget_.rightArmVelocity.data();
    case Robot_Control_Type::Left_Hand_Target:
        return isPosition ? finalTarget_.leftHandPosition.data() : finalTarget_.leftHandVelocity.data();
    case Robot_Control_Type::Right_Hand_Target:
        return isPosition ? finalTarget_.rightHandPosition.data() : finalTarget_.rightHandVelocity.data();
    case Robot_Control_Type::Neck_Target:
        return isPosition ? finalTarget_.neckPosition.data() : finalTarget_.neckVelocity.data();
    case Robot_Control_Type::Waist_Target:
        return isPosition ? finalTarget_.waistPosition.data() : finalTarget_.waistVelocity.data();
    default:
        return nullptr;
    }
}

void Robot_Control_Target::setNeckSendTarget_(bool planNew)
{
    auto type = Robot_Control_Type::Neck_Target;
    if (planNew)
        setGenericTarget_(type, getTargetPtr_(type, true), getTargetPtr_(type, false), NECK_JOINT_NUM, usePlanning_, true,
                          finalTargetValid_[static_cast<size_t>(type)]);
    else
        stepGenericTarget_(type, getTargetPtr_(type, true), getTargetPtr_(type, false), NECK_JOINT_NUM, usePlanning_, true,
                           finalTargetValid_[static_cast<size_t>(type)]);
}

void Robot_Control_Target::setArmSendTarget_(bool isLeft, bool planNew)
{
    size_t index = isLeft ? 2 : 2 + 7;
    auto controlType = isLeft ? Robot_Control_Type::Left_Arm_Target : Robot_Control_Type::Right_Arm_Target;
    bool &validFlag = finalTargetValid_[static_cast<size_t>(controlType)];
    auto &slowMoveJ = slowMoveJFlag_[isLeft ? 0 : 1];

    if (planNew)
    {
        setGenericTarget_(controlType, getTargetPtr_(controlType, true), getTargetPtr_(controlType, false), LEFT_ARM_JOINT_NUM,
                          usePlanning_ && !startRecvTargetsFlag_ && haveSetTarget_, true, validFlag, [&](size_t i, double val) {
                              if (startRecvTargetsFlag_ && targets_.targetBuffer.getDataValid(controlType))
                              {
                                  slowMoveJTarget_[isLeft ? 0 : 1](i) = static_cast<double>(val);
                                  targets_.targetManager.updateCurrentAcclLimitState(
                                      isLeft ? Robot_Control_Type::Left_Arm_Target : Robot_Control_Type::Right_Arm_Target, i, State(val));
                              }
                          });
        slowMoveJ = startRecvTargetsFlag_;
        if (targets_.targetBuffer.getDataValid(controlType))
            haveSetTarget_ = true;
    }
    else
        stepGenericTarget_(controlType, getTargetPtr_(controlType, true), getTargetPtr_(controlType, false), LEFT_ARM_JOINT_NUM,
                           usePlanning_ && !startRecvTargetsFlag_ && haveSetTarget_, true, validFlag);
}

void Robot_Control_Target::setHandSendTarget_(bool isLeft, bool planNew)
{
    size_t index = isLeft ? 2 + 7 + 7 : 2 + 7 + 7 + 6;
    auto controlType = isLeft ? Robot_Control_Type::Left_Hand_Target : Robot_Control_Type::Right_Hand_Target;
    if (planNew)
        setGenericTarget_(controlType, getTargetPtr_(controlType, true), getTargetPtr_(controlType, false), LEFT_HAND_JOINT_NUM,
                          usePlanning_ && handUsePlanning_, true, finalTargetValid_[static_cast<size_t>(controlType)]);
    else
        stepGenericTarget_(controlType, getTargetPtr_(controlType, true), getTargetPtr_(controlType, false), LEFT_HAND_JOINT_NUM,
                           usePlanning_ && handUsePlanning_, true, finalTargetValid_[static_cast<size_t>(controlType)]);
}

void Robot_Control_Target::setWaistSendTarget_(bool planNew)
{
    size_t index = 2 + 7 + 7 + 6 + 6;
    auto controlType = Robot_Control_Type::Waist_Target;
    if (planNew)
    {
        setGenericTarget_(controlType, getTargetPtr_(controlType, true), getTargetPtr_(controlType, false), WAIST_JOINT_NUM, usePlanning_,
                          true, finalTargetValid_[static_cast<size_t>(controlType)]);
    }
    else
        stepGenericTarget_(controlType, getTargetPtr_(controlType, true), getTargetPtr_(controlType, false), WAIST_JOINT_NUM, usePlanning_,
                           true, finalTargetValid_[static_cast<size_t>(controlType)]);
}

static inline double calculateVelocityFromPosition(double pos, double prevPos, double lastTime)
{
    return 0.9 * (pos - prevPos) / (getUnixTimestampInSeconds() - lastTime);
}

template <size_t N>
void Robot_Control_Target::fillAndWriteTarget_(const boost::array<double, N> &pos, const boost::array<double, N> &vel, size_t &bias,
                                               Robot_Control_Type controlType, size_t controlFlag)
{
    for (size_t i = 0; i < N; ++i)
    {
        formatedTarget_.at(2 * bias + i) = pos.at(i);
#if (defined ESTIMATE_VELOCITY_FROM_POSITION) && (ESTIMATE_VELOCITY_FROM_POSITION == 1)
        formatedTarget_.at(2 * bias + i + N) = calculateVelocityFromPosition(pos.at(i), targets_.targetBuffer.readData(controlType)[i],
                                                                             targets_.targetBuffer.getUpdateTime(controlType));
#else
        formatedTarget_.at(2 * bias + i + N) = vel.at(i);
#endif
    }
    targets_.targetBuffer.writeNewData(controlType, &formatedTarget_.at(2 * bias), flags_.getControlValidFlag(controlFlag, controlType));
    bias += N;
}

template <size_t N>
void Robot_Control_Target::fillAndWriteTarget_(const double *pos, const double *vel, size_t &bias, Robot_Control_Type controlType,
                                               size_t controlFlag)
{
    for (size_t i = 0; i < N; ++i)
    {
        formatedTarget_.at(2 * bias + i) = pos[i];
#if (defined ESTIMATE_VELOCITY_FROM_POSITION) && (ESTIMATE_VELOCITY_FROM_POSITION == 1)
        formatedTarget_.at(2 * bias + i + N) = calculateVelocityFromPosition(pos[i], targets_.targetBuffer.readData(controlType)[i],
                                                                             targets_.targetBuffer.getUpdateTime(controlType));
#else
        formatedTarget_.at(2 * bias + i + N) = vel[i];
#endif
    }
    targets_.targetBuffer.writeNewData(controlType, &formatedTarget_.at(2 * bias), flags_.getControlValidFlag(controlFlag, controlType));
    bias += N;
}

void Robot_Control_Target::setNewTarget(const robot_uplimb_pkg::WholeBodyPositionVelocityConstPtr &target, size_t controlFlag)
{
    if (getUnixTimestampInSeconds() - lastRecvTargetTime_ > 0.5)
        startRecvTargetsFlag_ = true;
    lastRecvTargetTime_ = getUnixTimestampInSeconds();
    size_t bias = 0;
    fillAndWriteTarget_(target->neckPosition, target->neckVelocity, bias, Robot_Control_Type::Neck_Target, controlFlag);
    fillAndWriteTarget_<LEFT_ARM_JOINT_NUM>(target->leftArmPosition.data(), target->leftArmVelocity.data(), bias,
                                            Robot_Control_Type::Left_Arm_Target, controlFlag);
    fillAndWriteTarget_<RIGHT_ARM_JOINT_NUM>(target->rightArmPosition.data(), target->rightArmVelocity.data(), bias,
                                             Robot_Control_Type::Right_Arm_Target, controlFlag);
    fillAndWriteTarget_(target->leftHandPosition, target->leftHandVelocity, bias, Robot_Control_Type::Left_Hand_Target, controlFlag);
    fillAndWriteTarget_(target->rightHandPosition, target->rightHandVelocity, bias, Robot_Control_Type::Right_Hand_Target, controlFlag);
    fillAndWriteTarget_<WAIST_JOINT_NUM>(target->waistPosition.data(), target->waistVelocity.data(), bias, Robot_Control_Type::Waist_Target,
                                         controlFlag);
}

void Robot_Control_Target::updateNewTarget(const std::array<bool, static_cast<size_t>(Robot_Control_Type::Target_All)> &updateFlags)
{
    if (!allowMoving_)
        return;
#if (defined TEST_ARM_SINGLE_JOINT_MPC) && (TEST_ARM_SINGLE_JOINT_MPC == 0)
    if (updateFlags.at(static_cast<size_t>(Robot_Control_Type::Neck_Target)))
        setNeckSendTarget_(true);
#endif
    if (updateFlags.at(static_cast<size_t>(Robot_Control_Type::Left_Arm_Target)))
        setArmSendTarget_(true, true);
#if (defined TEST_ARM_SINGLE_JOINT_MPC) && (TEST_ARM_SINGLE_JOINT_MPC == 0)
    if (updateFlags.at(static_cast<size_t>(Robot_Control_Type::Right_Arm_Target)))
        setArmSendTarget_(false, true);
    if (updateFlags.at(static_cast<size_t>(Robot_Control_Type::Waist_Target)))
        setWaistSendTarget_(true);
    if (updateFlags.at(static_cast<size_t>(Robot_Control_Type::Left_Hand_Target)))
        setHandSendTarget_(true, true);
    if (updateFlags.at(static_cast<size_t>(Robot_Control_Type::Right_Hand_Target)))
        setHandSendTarget_(false, true);
#endif
    haveCalculatedMPC_ = true;
}

void Robot_Control_Target::mpcPlannerCheckUpdateTime_(void)
{
    if (!allowMoving_)
        return;
#if (defined TEST_ARM_SINGLE_JOINT_MPC) && (TEST_ARM_SINGLE_JOINT_MPC == 0)
    setNeckSendTarget_(false);
#endif
    setArmSendTarget_(true, false);
#if (defined TEST_ARM_SINGLE_JOINT_MPC) && (TEST_ARM_SINGLE_JOINT_MPC == 0)
    setArmSendTarget_(false, false);
    setWaistSendTarget_(false);
    setHandSendTarget_(true, false);
    setHandSendTarget_(false, false);
#endif
}

const robot_uplimb_pkg::WholeBodyPositionVelocity &Robot_Control_Target::getCurrentNewTarget(void)
{
    static bool stashedStartFlag = true;
    static size_t startupRecheckCNT = 0;
    stashedStartFlag = startRecvTargetsFlag_;
    if (!allowMoving_)
        return finalTarget_;
    if (getUnixTimestampInSeconds() - lastRecvTargetTime_ > 1.0)
        LOG_WARN() << "Receive target timeout.";
    if (useMPC_)
        mpcPlannerCheckUpdateTime_();
    if (stashedStartFlag && startRecvTargetsFlag_ && haveSetTarget_)
    {
#if (defined TEST_ARM_SINGLE_JOINT_MPC) && (TEST_ARM_SINGLE_JOINT_MPC == 0)
        if (checkArmControlTargetError(true) && checkArmControlTargetError(false))
#endif
        {
            startupRecheckCNT = (startupRecheckCNT + 1) % 60;
            if (startupRecheckCNT == 0)
                startRecvTargetsFlag_ = false;
        }
    }
    else
        startupRecheckCNT = 0;
    return finalTarget_;
}

std::array<double, 12> &Robot_Control_Target::getCurrentForceFeedback(void)
{
    return forceFeedback_;
}

void Robot_Control_Target::setPlanTime(double planTime)
{
    planTime_ = planTime;
}

bool Robot_Control_Target::getControlFlag(Robot_Control_Type controlType) const
{
    return flags_.getControlFlag(controlType);
}

bool Robot_Control_Target::requireJointStatesHaveReceived(void) const
{
    return jointStatesReceived_;
}

bool Robot_Control_Target::requireForcesHaveReceived(void) const
{
    return forceReceived_;
}

const robot_uplimb_pkg::WholeBodyPositionVelocity &Robot_Control_Target::getNewJointStates(void) const
{
    return newJointState_;
}

const std::array<bool, static_cast<size_t>(Robot_Control_Type::Target_All)> &Robot_Control_Target::getTargetValidation(void) const
{
    return finalTargetValid_;
}

bool Robot_Control_Target::getTargetValidation(Robot_Control_Type controlType) const
{
    return finalTargetValid_[static_cast<size_t>(controlType)];
}

const bool *Robot_Control_Target::getSlowMoveJFlag(void) const
{
    return reinterpret_cast<const bool *>(slowMoveJFlag_);
}

bool Robot_Control_Target::getMovePermission(void) const
{
    return allowMoving_;
}
