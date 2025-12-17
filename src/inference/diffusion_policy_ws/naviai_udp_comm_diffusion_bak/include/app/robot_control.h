#ifndef ROBOT_CONTROL_H
#define ROBOT_CONTROL_H

#include "accl_limit/accl_limit_module.h"
#include "control_target_buffer/control_target_buffer.hpp"
#include "utils.h"

#include "robot_uplimb_pkg/WholeBodyPositionVelocity.h"

#include <Eigen/Dense>
#include <array>
#include <boost/array.hpp>
#include <cstddef>
#include <stdexcept>
#include <type_traits>

typedef enum : std::uint8_t
{
    Test_Send_Image = 0,
    Send_Joint_Feedback = 1,
    Send_Image_Feedback,
    Send_Force_Feedback,
    Feedback_Tasks_All
} Feedback_Tasks_e;

class ControlTargetManager
{
  public:
    ControlTargetManager(const std::vector<Accl_Limit_Config> &configs, const std::vector<State> &initialState, double planTime,
                         bool useMPC = true);
    ControlTargetManager(const Accl_Limit_Config &config, size_t dof, double planTime, bool useMPC = true);
    virtual ~ControlTargetManager();
    size_t getSize(void);
    void setLimits(size_t index, const Accl_Limit_Config &config);
    void setLimits(const std::vector<Accl_Limit_Config> &configs);
    void updateCurrentAcclLimitState(size_t index, const State &state);
    void updateCurrentAcclLimitState(const std::vector<State> &states);
    void updateNewTarget(size_t index, const State &target);
    void updateNewTarget(const std::vector<State> &targets);
    void updateNewTarget(const double *const positionTarget, const double *const speedTarget, const double *const accelerationTarget,
                         const double *const jerkTarget);
    void updatePlanTime(double planTime);
    const std::vector<State> planNewTarget(void);
    const std::vector<State> planNewTarget(const std::vector<State> &targets);
    const std::vector<State> planNewTarget(const double *const positionTarget, const double *const speedTarget,
                                           const double *const accelerationTarget, const double *const jerkTarget);

    const std::vector<Eigen::VectorXd> &getLastPlannedControlVector(void) const;
    double getLastSmoothTime(void) const;
    double getLastStepTime(void) const;
    const std::vector<State> stepPlannedTarget(const std::vector<double> &control, double dt);
    const std::vector<State> stepPlannedTarget(size_t lastOutputIndex, double dt);

  private:
    Accl_Limit_Smoother smoother_;
    std::vector<State> targets_;
    double planTime_, lastSmoothTime_, lastSteppedTime_;
    size_t size_;

    static constexpr bool qpVerbose_ = true; // Set to true for debugging purposes, false for normal operation
};

class ControlTargetManagers
{
  public:
    ControlTargetManagers(double dt, bool useMPC = true);
    virtual ~ControlTargetManagers();
    void setLimits(Robot_Control_Type type, size_t index, const Accl_Limit_Config &config);
    void setLimits(Robot_Control_Type type, const std::vector<Accl_Limit_Config> &configs);
    void updateCurrentAcclLimitState(Robot_Control_Type type, size_t index, const State &state);
    void updateCurrentAcclLimitState(Robot_Control_Type type, const std::vector<State> &states);
    const std::vector<State> planNewTarget(Robot_Control_Type type, const std::vector<State> &targets, double planTime);
    const std::vector<State> planNewTarget(Robot_Control_Type type, const double *const positionTarget, const double *const speedTarget,
                                           const double *const accelerationTarget, const double *const jerkTarget, double planTime);
    const std::vector<Eigen::VectorXd> &getLastPlannedControlVector(Robot_Control_Type type);
    double getLastSmoothTime(Robot_Control_Type type);
    double getLastStepTime(Robot_Control_Type type);
    const std::vector<State> stepPlannedTarget(Robot_Control_Type type, const std::vector<double> &control, double dt);
    const std::vector<State> stepPlannedTarget(Robot_Control_Type type, size_t lastOutputIndex, double dt);

  private:
    ControlTargetManager neckTargetManager_;
    std::array<ControlTargetManager, 2> armTargetManager_;
    std::array<ControlTargetManager, 2> handTargetManager_;
    ControlTargetManager waistTargetManager_;

    ControlTargetManager &getManager_(Robot_Control_Type type);
};

class ControlTargets
{
  public:
    ControlTargets(bool useDoubleBuffer, double dt, bool useMPC = true);
    virtual ~ControlTargets();

    ControlTargetBuffer<double> targetBuffer;
    ControlTargetManagers targetManager;
};

class Robot_Control_Target
{
  public:
    Robot_Control_Target(bool useDoubleBuffer, bool handUsePlan, bool useMPC);
    virtual ~Robot_Control_Target();

    void setPlanTime(double planTime);
    bool getControlFlag(Robot_Control_Type controlType) const;
    void setMovingPermissionFlag(bool permission);
    bool getMovePermission(void) const;
    void updateNewJointState(const std::array<double, BODY_JOINT_NUM> &position, const std::array<double, BODY_JOINT_NUM> &velocity);
    void updateNewForceFeedback(const Eigen::Vector3d &force, const Eigen::Vector3d &torque, bool isLeft);
    const robot_uplimb_pkg::WholeBodyPositionVelocity &getNewJointStates(void) const;
    void setNewTarget(const robot_uplimb_pkg::WholeBodyPositionVelocityConstPtr &target, size_t controlFlag);
    void updateNewTarget(const std::array<bool, static_cast<size_t>(Robot_Control_Type::Target_All)> &updateFlags =
                             std::array<bool, static_cast<size_t>(Robot_Control_Type::Target_All)>({true, true, true, true, true, true}));
    const robot_uplimb_pkg::WholeBodyPositionVelocity &getCurrentNewTarget(void);
    std::array<double, 12> &getCurrentForceFeedback(void);
    bool checkArmControlTargetError(bool isLeft);
    bool requireJointStatesHaveReceived(void) const;
    bool requireForcesHaveReceived(void) const;
    const std::array<bool, static_cast<size_t>(Robot_Control_Type::Target_All)> &getTargetValidation(void) const;
    bool getTargetValidation(Robot_Control_Type controlType) const;
    const bool *getSlowMoveJFlag(void) const;
    bool getFeedbackTasksFlag(Feedback_Tasks_e task) const;

  private:
    static constexpr double mpcStepTime_ = 1 / 50.0;
    static constexpr double stepTime_ = 1 / 240.0;

    bool useMPC_, usePlanning_, handUsePlanning_, allowMoving_, haveSetTarget_, jointStatesReceived_, forceReceived_;
    volatile bool startRecvTargetsFlag_, haveCalculatedMPC_;
    double planTime_, jointStatesReceivedTime_, forceReceivedTime_, lastRecvTargetTime_;
    robot_uplimb_pkg::WholeBodyPositionVelocity newJointState_;
    HandleControlFlags flags_;
    ControlTargets targets_;
    robot_uplimb_pkg::WholeBodyPositionVelocity finalTarget_;
    std::array<double, JOINT_NUM * 2> formatedTarget_;
    std::array<double, 12> forceFeedback_;
    std::array<bool, static_cast<size_t>(Robot_Control_Type::Target_All)> finalTargetValid_;
    bool slowMoveJFlag_[2];
    std::array<Eigen::VectorXd, 2> slowMoveJTarget_;
    std::array<bool, static_cast<size_t>(Feedback_Tasks_e::Feedback_Tasks_All)> feedbackTasks_;
    Accl_Limit_Config acclLimitParams_;
    std::string packagePath_;

    void loadConfig_(void);
    void printConfig_(YAML::Node config) const;
    void loadAndSetJointTargets_(Robot_Control_Type controlType, size_t offset, size_t count, YAML::Node &robotJointLimits);
    void loadJointLimits_(void);
    double *const getTargetPtr_(Robot_Control_Type type, bool isPosition);
    void setPlanCurrentPointFromJointStates_(void);
    void setGenericTarget_(Robot_Control_Type controlType, double *const positionTarget, double *const velocityTarget, size_t dof,
                           bool usePlan, bool needJointState, bool &validFlag,
                           std::function<void(size_t, double)> optionalCallback = nullptr);
    void stepGenericTarget_(Robot_Control_Type controlType, double *const positionTarget, double *const velocityTarget, size_t dof,
                            bool usePlan, bool needJointState, bool &validFlag);
    void mpcPlannerCheckUpdateTime_(void);
    void setNeckSendTarget_(bool planNew);
    void setArmSendTarget_(bool isLeft, bool planNew);
    void setHandSendTarget_(bool isLeft, bool planNew);
    void setWaistSendTarget_(bool planNew);
    template <size_t N>
    void fillAndWriteTarget_(const boost::array<double, N> &pos, const boost::array<double, N> &vel, size_t &bias,
                             Robot_Control_Type controlType, size_t controlFlag);
    template <size_t N>
    void fillAndWriteTarget_(const double *pos, const double *vel, size_t &bias, Robot_Control_Type controlType, size_t controlFlag);
};

#endif /* ROBOT_CONTROL_H */
