#include "app/naviai_app.h"
#include "app/robot_control.h"
#include "ros/init.h"
#include "utils.h"

#include <atomic>
#include <boost/process.hpp>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <stdexcept>
#include <string>
#include <thread>

#include <ros/package.h>
#include <ros/ros.h>

#include <geometry_msgs/Twist.h>
#include <geometry_msgs/WrenchStamped.h>
#include <sensor_msgs/JointState.h>
#include <std_msgs/Bool.h>
#include <std_msgs/Float32MultiArray.h>
#include <std_srvs/Empty.h>

#include "navi_types/MoveJ.h"
#include "navi_types/RobotHandJointSrv.h"
#include "naviai_udp_comm/TeleoperationUDPRaw.h"
#include "robot_uplimb_pkg/TeleoperationTarget.h"
#include "robot_uplimb_pkg/WholeBodyPositionVelocity.h"

#define TEST_MODE 1
#define TEST_DIFFUSION 1

#define QOS_FREQ_FILTER_RATE 0.03
#define QOS_FREQ_THRESHOLD 0.6

#define TEST_SEND_TO_WHOLE_ROBOT_CTRL false

// #define CALL_WHEELED_WAIST_MOVE_SERVICE
#undef CALL_WHEELED_WAIST_MOVE_SERVICE

#define NODE_NAME "teleoperation_communication_robot_node"

class Naviai_UDP_Communicate_Robot : private Naviai_UDP_Communicate_App, public Robot_Control_Target
{
  public:
    Naviai_UDP_Communicate_Robot(bool useMPC)
        : Naviai_UDP_Communicate_App(NODE_NAME, 28257, 28258, 28256, true), Robot_Control_Target(true, false, useMPC),
          lastRecvTargetTick_(getUnixTimestampInSeconds()), firstRecvTargetFlag_(true), useMPC_(useMPC),
          controlWheeledRobotChassisWaistUsingController_(false), useRobotTypeName_("H1_Pro"), mpcThread1Ready_(false),
          mpcThread2Ready_(false), recordServiceReadyFlag_(false), waistHaveControlledFlag_(false)
    {
        lastWaistTarget_.fill(0);
        initROSNode_();
    }

    ~Naviai_UDP_Communicate_Robot()
    {
        Naviai_UDP_Communicate_App::stop();
    }

    void start(void)
    {
        lastRecvTargetTick_ = getUnixTimestampInSeconds();
        std::thread(&Naviai_UDP_Communicate_Robot::controlRobotBodyThread_, this).detach();
        if (getControlFlag(Robot_Control_Type::Left_Hand_Target))
            std::thread(&Naviai_UDP_Communicate_Robot::controlRobotHandThread_, this).detach();
#if (defined USING_NAVIAI_ROBOT_WA1_0303) || (defined USING_NAVIAI_ROBOT_WA2_A2_LITE)
        if (controlWheeledRobotChassisWaistUsingController_)
        {
            std::thread(&Naviai_UDP_Communicate_Robot::controlChassisThread_, this).detach();
#if defined CALL_WHEELED_WAIST_MOVE_SERVICE
            std::thread(&Naviai_UDP_Communicate_Robot::controlWheeledWaistFromVRThread_, this).detach();
#endif
        }
#endif
        if (useMPC_)
        {
            std::thread(&Naviai_UDP_Communicate_Robot::calculateMPCTimerThread_, this).detach();
            std::thread(&Naviai_UDP_Communicate_Robot::calculateMPCThread_, this, true).detach();
            std::thread(&Naviai_UDP_Communicate_Robot::calculateMPCThread_, this, false).detach();
        }
        std::thread(&Naviai_UDP_Communicate_Robot::recordRosbagThread_, this).detach();
        if (getFeedbackTasksFlag(Feedback_Tasks_e::Send_Force_Feedback))
            std::thread(&Naviai_UDP_Communicate_Robot::sendForceFeedbackThread_, this).detach();
        Naviai_UDP_Communicate_App::start();
        ros::spin();
    }

  private:
    double stepFrequency_, cmdFrequency_;
    bool useNAVIAIWholeRobotSendCtrl_, controlWheeledRobotChassisWaistUsingController_, useMPC_;
    std::string packagePath_, useRobotTypeName_;

    size_t waistCtrlServiceJointSize_;

    std::atomic<bool> recordDataFlag_{false}, waistMoveJMoving_{false};
    std::array<double, BODY_JOINT_NUM> recvJointPositions_, recvJointVeclocities_;

    ros::ServiceClient controlHandServiceClient_;
    std::array<ros::ServiceClient, 2> recordServiceClient_;
    bool recordServiceReadyFlag_, waistHaveControlledFlag_;
    std::array<double, WAIST_JOINT_NUM> lastWaistTarget_;
    std::timed_mutex recordServiceMutex_;
    std::array<ros::Publisher, 2> teleoperationSetTargetPublishers_;
    ros::Publisher teleoperationRecordDataStatePublisher_;
#if (TEST_MODE == 0)
    ros::Publisher teleoperationRecvTargetPublisher_;
#else
    ros::Subscriber teleoperationRecvTargetSubscriber_;
#endif
    ros::Subscriber jointStatesSubscriber_;
    std::array<ros::Subscriber, 2> forceSubscribers_;
#if (defined USING_NAVIAI_ROBOT_WA1_0303) || (defined USING_NAVIAI_ROBOT_WA2_A2_LITE)
    ros::Publisher teleoperationControlChassisPublisher_;
    ros::ServiceClient teleoperationControlWheeledWaistFromVRClient_;
#endif

    naviai_udp_comm::TeleoperationUDPRaw recvRawTarget_;
    robot_uplimb_pkg::TeleoperationTarget finalTarget_;
    geometry_msgs::Twist chassisCmd_;
    navi_types::MoveJ teleoperationControlWheeledWaistSRV_;

    bool firstRecvTargetFlag_;
    double realCalculatedFPS_, lastRecvTargetTick_;

    std::mutex mpcCalculateMutex_;
    std::condition_variable mpcCalculateCV_;
    bool mpcThread1Ready_, mpcThread2Ready_;

    void initROSNode_(void) override
    {
        packagePath_ = ros::package::getPath("naviai_udp_comm");
        stepFrequency_ = getParam_<double>("/step_frequency", 300.0);
        cmdFrequency_ = getParam_<double>("/command_frequency", 300);
        realCalculatedFPS_ = cmdFrequency_;
        setPlanTime(1.0 / stepFrequency_ * 1.02);
        useRobotTypeName_ = getParam_<std::string>("/use_robot_type", "");
        if (useRobotTypeName_ == "H1_Pro")
            waistCtrlServiceJointSize_ = 1;
        else if (useRobotTypeName_ == "WA1_0303")
            waistCtrlServiceJointSize_ = 2;
        else if (useRobotTypeName_ == "WA2_A2_lite")
            waistCtrlServiceJointSize_ = 4;
        else
        {
            ROS_FATAL("Expected robot type of H1_Pro/WA1_0303/WA2_A2_lite, get invalid parameter of %s.", useRobotTypeName_.c_str());
            throw std::runtime_error("Invalid robot type.");
        }
        useNAVIAIWholeRobotSendCtrl_ = TEST_SEND_TO_WHOLE_ROBOT_CTRL;
        controlWheeledRobotChassisWaistUsingController_ = getParam_<bool>("/control_wheeled_robot_move", false);
        if (packagePath_.empty())
        {
            ROS_ERROR("Failed to find package naviai_udp_comm.");
            return;
        }
#if (TEST_MODE == 0) || (TEST_DIFFUSION == 1)
        if (getControlFlag(Robot_Control_Type::Left_Hand_Target))
        {
            ROS_INFO("Start to wait for Hand control services.");
            ros::service::waitForService("/robotHandJointSwitch");
            controlHandServiceClient_ = nh.serviceClient<navi_types::RobotHandJointSrv>("/robotHandJointSwitch");
            ROS_INFO("Service is available.");
        }
#endif
#if (defined USING_NAVIAI_ROBOT_WA1_0303) || (defined USING_NAVIAI_ROBOT_WA2_A2_LITE)
        if (controlWheeledRobotChassisWaistUsingController_)
        {
            ROS_INFO("Started to wait for waist service and initialize chassis control interface.");
            teleoperationControlChassisPublisher_ = nh.advertise<geometry_msgs::Twist>("/calib_vel", 10);
            ros::service::waitForService("/waist_movej_service");
            teleoperationControlWheeledWaistFromVRClient_ = nh.serviceClient<navi_types::MoveJ>("/waist_movej_service");
            ROS_INFO("Waist control service is available.");
        }
        else
            ROS_INFO("Skipping configuration for waist and chassis control handle.");
#else
        ROS_INFO("Not wheeled robot, no need to control chassis and waist.");
#endif
#if (TEST_MODE == 0) || (TEST_DIFFUSION == 1)
        teleoperationSetTargetPublishers_[0] = nh.advertise<robot_uplimb_pkg::TeleoperationTarget>("/teleoperation_ctrl_cmd/final", 10);
#else
        teleoperationSetTargetPublishers_[0] =
            nh.advertise<robot_uplimb_pkg::TeleoperationTarget>("/teleoperation_ctrl_cmd/final_test", 10);
#endif
        teleoperationSetTargetPublishers_[1] = nh.advertise<robot_uplimb_pkg::TeleoperationTarget>("/teleoperation_ctrl_cmd/send_ctrl", 10);
        teleoperationRecordDataStatePublisher_ = nh.advertise<std_msgs::Bool>("/teleoperation_ctrl_cmd/record_data", 10);
#if (TEST_MODE == 0)
        teleoperationRecvTargetPublisher_ = nh.advertise<naviai_udp_comm::TeleoperationUDPRaw>("/teleoperation_ctrl_cmd/recv_raw", 10);
#else
        teleoperationRecvTargetSubscriber_ =
            nh.subscribe("/teleoperation_ctrl_cmd/recv_raw", 10, &Naviai_UDP_Communicate_Robot::unpackTestMode_, this);
#endif
        jointStatesSubscriber_ = nh.subscribe("/joint_states", 10, &Naviai_UDP_Communicate_Robot::onJointStatesReceive_, this);
        forceSubscribers_.at(0) = nh.subscribe<geometry_msgs::WrenchStamped>(
            "/force_sensor_6D_left", 10, boost::bind(&Naviai_UDP_Communicate_Robot::onForceFeedbackReceive_, this, _1, true));
        forceSubscribers_.at(1) = nh.subscribe<geometry_msgs::WrenchStamped>(
            "/force_sensor_6D_right", 10, boost::bind(&Naviai_UDP_Communicate_Robot::onForceFeedbackReceive_, this, _1, false));
        teleoperationControlWheeledWaistSRV_.request.jnt_angle.resize(waistCtrlServiceJointSize_);
        std::thread([&]() -> void {
            ROS_INFO("Waiting for record service /hdf5/start_recording and /hdf5/stop_recording");
            ros::service::waitForService("/hdf5/start_recording");
            ros::service::waitForService("/hdf5/stop_recording");
            recordServiceClient_.at(0) = nh.serviceClient<std_srvs::Empty>("/hdf5/start_recording");
            recordServiceClient_.at(1) = nh.serviceClient<std_srvs::Empty>("/hdf5/stop_recording");
            recordServiceReadyFlag_ = true;
            ROS_INFO("Get record service /hdf5/start_recording and /hdf5/stop_recording");
        }).detach();
    }

    void updateRecvTargetFreq_(void)
    {
        if ((!firstRecvTargetFlag_) && (getUnixTimestampInSeconds() - lastRecvTargetTick_ > 1e-4))
            realCalculatedFPS_ = realCalculatedFPS_ * (1 - QOS_FREQ_FILTER_RATE) +
                                 1.0 / (getUnixTimestampInSeconds() - lastRecvTargetTick_) * QOS_FREQ_FILTER_RATE;
        lastRecvTargetTick_ = getUnixTimestampInSeconds();
        firstRecvTargetFlag_ = false;
        if (realCalculatedFPS_ < cmdFrequency_ * QOS_FREQ_THRESHOLD)
            ROS_ERROR("Communication quality bad! Expected %lf fps, get %lf fps.", cmdFrequency_, realCalculatedFPS_);
    }

    void onUDPDataRecv_(std::vector<std::uint8_t> &&data, NAVIAI_UDP_Communication::Packet_Type_Enum type, std::uint64_t sendTimestamp,
                        std::uint64_t recvTimestamp, const std::string &sender_ip, int sender_port) override
    {
        if (!isRunning())
            return;
        switch (type)
        {
        case NAVIAI_UDP_Communication::Packet_Type_Enum::Test_Latency:
            break;
        case NAVIAI_UDP_Communication::Packet_Type_Enum::Packed_Target:
            updateRecvTargetFreq_();
            unpackAll_(std::move(data));
            break;
        default:
            break;
        }
    }

    void onForceFeedbackReceive_(const geometry_msgs::WrenchStampedConstPtr &msg, bool isLeft)
    {
        try
        {
            if (!isRunning())
                return;
            const auto &f = msg->wrench.force;
            const auto &t = msg->wrench.torque;
            updateNewForceFeedback(Eigen::Vector3d(f.x, f.y, f.z), Eigen::Vector3d(t.x, t.y, t.z), isLeft);
        }
        catch (const std::exception &e)
        {
            ROS_ERROR("Get error %s while receiving force feedback.", e.what());
        }
    }

    void sendForceFeedbackThread_(void)
    {
#if (TEST_MODE == 0)
        static size_t errorCNT = 0;
        std::this_thread::sleep_for(std::chrono::seconds(3));
        ros::Rate runRate(stepFrequency_);
        while (isRunning() && ros::ok())
        {
            runRate.sleep();
            if (requireForcesHaveReceived())
            {
                auto res = packSendData(reinterpret_cast<const std::uint8_t *>(getCurrentForceFeedback().data()),
                                        getCurrentForceFeedback().size() * 4, NAVIAI_UDP_Communication::Packet_Type_Enum::Force_Feedback);
                for (auto r : res)
                    if (r <= 0)
                        ROS_ERROR("Send data failed.");
            }
            else
            {
                errorCNT = (errorCNT + 1) % 0x100000;
                if ((errorCNT % 100) == 0)
                    ROS_WARN("No force feedback received.");
            }
        }
#endif
    }

    void onJointStatesReceive_(const sensor_msgs::JointStateConstPtr &msgptr)
    {
        try
        {
            if (!isRunning())
                return;
            const sensor_msgs::JointState &msg = *msgptr;
            bool positionValid = (msg.position.size() == msg.name.size()), velocityValid = (msg.velocity.size() == msg.name.size());
            if ((!positionValid) || (!velocityValid))
            {
                ROS_WARN("Invalid joint_states msg, get position of size %ld and velocity of size %ld.", msg.position.size(),
                         msg.velocity.size());
                if ((!positionValid) && (!velocityValid))
                    return;
            }
            size_t foundJointCNT = 0;
            for (size_t i = 0; i < msg.name.size(); ++i)
            {
                const std::string &name = msg.name[i];
                auto index = jointName2IndexMap.find(name);
                if (index != jointName2IndexMap.end())
                {
                    size_t jointIndex = index->second;
                    if (jointIndex < BODY_JOINT_NUM)
                    {
                        if (positionValid)
                            recvJointPositions_[jointIndex] = msg.position[i];
                        if (velocityValid)
                            recvJointVeclocities_[jointIndex] = msg.velocity[i];
                    }
                    else
                    {
                        ROS_WARN("Invalid joint index: %zu for joint '%s'.", jointIndex, name.c_str());
                        continue;
                    }
                    foundJointCNT += 1;
                }
                else
                {
                    ROS_WARN("Get msg have invalid joint '%s'.", name.c_str());
                    continue;
                }
                if ((i == msg.name.size() - 1) || (foundJointCNT == BODY_JOINT_NUM))
                {
                    if (!getFeedbackTasksFlag(Feedback_Tasks_e::Send_Joint_Feedback))
                        return;
                    updateNewJointState(recvJointPositions_, recvJointVeclocities_);
                    auto res = packSendData(serializeRosMessage(getNewJointStates()),
                                            NAVIAI_UDP_Communication::Packet_Type_Enum::Joint_States_Feedback);
                    for (auto r : res)
                        if (r <= 0)
                            ROS_ERROR("Send data failed.");
                }
            }
        }
        catch (const std::exception &e)
        {
            ROS_ERROR("Get error %s while receiving /joint_states topic.", e.what());
        }
    }

    template <typename T> static bool validateAndCopyTarget_(float src, T &dst, const std::string &name)
    {
        if (getNumberIsNormal(src))
        {
            dst = static_cast<T>(src);
            return true;
        }
        else
        {
            ROS_ERROR("Get invalid %s target %f.", name.c_str(), src);
            return false;
        }
    }

    void unpackAll_(std::vector<std::uint8_t> &&data)
    {
#if (TEST_MODE == 0)
        if (!isRunning())
            return;
        if (!deserializeRosMessage(data, recvRawTarget_))
        {
            ROS_ERROR("Get error while unpacking target from teleoperation server.");
            return;
        }
        teleoperationRecvTargetPublisher_.publish(recvRawTarget_);
        for (size_t i = 0; i < waistCtrlServiceJointSize_; ++i)
            teleoperationControlWheeledWaistSRV_.request.jnt_angle.at(i) = recvRawTarget_.target.waistTargetForWheeled.at(i);
        chassisCmd_ = recvRawTarget_.target.chassisCmd;
        recordDataFlag_.store(recvRawTarget_.recordingFlag);
        setNewTarget(boost::make_shared<robot_uplimb_pkg::WholeBodyPositionVelocity>(recvRawTarget_.target.target),
                     recvRawTarget_.controlFlag);
#endif
    }

#if (TEST_MODE == 1)
    void unpackTestMode_(const naviai_udp_comm::TeleoperationUDPRawConstPtr &msg)
    {
        if (!isRunning())
            return;
        for (size_t i = 0; i < waistCtrlServiceJointSize_; ++i)
            teleoperationControlWheeledWaistSRV_.request.jnt_angle.at(i) = msg->target.waistTargetForWheeled.at(i);
        chassisCmd_ = msg->target.chassisCmd;
        recordDataFlag_.store(msg->recordingFlag);
        setNewTarget(boost::make_shared<robot_uplimb_pkg::WholeBodyPositionVelocity>(msg->target.target), msg->controlFlag);
    }
#endif

    void calculateMPCTimerThread_(void)
    {
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
        ros::Rate runRate(50);
        while (isRunning() && ros::ok())
        {
            try
            {
                runRate.sleep();
                {
                    std::lock_guard<std::mutex> lk(mpcCalculateMutex_);
                    mpcThread1Ready_ = true;
                    mpcThread2Ready_ = true;
                    mpcCalculateCV_.notify_all();
                }
            }
            catch (const std::exception &e)
            {
                ROS_ERROR("Get error %s while setting timers for robot body MPC calculation thread.", e.what());
            }
        }
    }

    void calculateMPCThread_(bool isFirstThread)
    {
        bool &threadReady = isFirstThread ? mpcThread1Ready_ : mpcThread2Ready_;
        const std::array<bool, static_cast<size_t>(Robot_Control_Type::Target_All)> updateFlags =
            isFirstThread ? std::array<bool, static_cast<size_t>(Robot_Control_Type::Target_All)>({true, true, false, true, false, false})
                          : std::array<bool, static_cast<size_t>(Robot_Control_Type::Target_All)>({false, false, true, false, true, true});
        std::this_thread::sleep_for(std::chrono::milliseconds(800));
        while (isRunning() && ros::ok())
        {
            try
            {
                {
                    std::unique_lock<std::mutex> lk(mpcCalculateMutex_);
                    mpcCalculateCV_.wait(lk, [&threadReady] { return threadReady; });
                    threadReady = false;
                }
                updateNewTarget(updateFlags);
            }
            catch (const std::exception &e)
            {
                ROS_ERROR("Get error %s while calculating target angles for robot body in thead %d.", e.what(),
                          static_cast<int>(isFirstThread));
            }
        }
    }

    void controlRobotBodyThread_(void)
    {
        std::this_thread::sleep_for(std::chrono::seconds(2));
        ros::Rate runRate(stepFrequency_);
        ros::Time startTime;
        ROS_INFO("Start sending target thread.");
        while (isRunning() && ros::ok())
        {
            runRate.sleep();
            if (!getMovePermission())
                continue;
            try
            {
                if (!useMPC_)
                    updateNewTarget();
                finalTarget_.target = getCurrentNewTarget();
                finalTarget_.neckValid = getTargetValidation(Robot_Control_Type::Neck_Target);
                finalTarget_.leftArmValid = getTargetValidation(Robot_Control_Type::Left_Arm_Target);
                finalTarget_.leftHandValid = getTargetValidation(Robot_Control_Type::Left_Hand_Target);
                finalTarget_.rightArmValid = getTargetValidation(Robot_Control_Type::Right_Arm_Target);
                finalTarget_.rightHandValid = getTargetValidation(Robot_Control_Type::Right_Hand_Target);
                finalTarget_.waistValid = getTargetValidation(Robot_Control_Type::Waist_Target);
#if ((defined USING_NAVIAI_ROBOT_WA1_0303) || (defined USING_NAVIAI_ROBOT_WA2_A2_LITE)) && (!(defined CALL_WHEELED_WAIST_MOVE_SERVICE))
                bool waistTargetValid = checkWaistErrorValid_(teleoperationControlWheeledWaistSRV_.request.jnt_angle.data());
                if (!waistTargetValid)
                    std::thread(&Naviai_UDP_Communicate_Robot::moveWaistFirstPoint_, this).detach();
                finalTarget_.waistValid = waistTargetValid && (!waistMoveJMoving_.load());
                for (size_t i = 0; i < teleoperationControlWheeledWaistSRV_.request.jnt_angle.size(); ++i)
                    finalTarget_.target.waistPosition.at(i) = teleoperationControlWheeledWaistSRV_.request.jnt_angle.at(i);
                if (finalTarget_.waistValid)
                    for (size_t i = 0; i < teleoperationControlWheeledWaistSRV_.request.jnt_angle.size(); ++i)
                        lastWaistTarget_.at(i) = teleoperationControlWheeledWaistSRV_.request.jnt_angle.at(i);
#endif
                finalTarget_.leftArmSlowMoveJFlag = getSlowMoveJFlag()[0];
                finalTarget_.rightArmSlowMoveJFlag = getSlowMoveJFlag()[1];
                if (requireJointStatesHaveReceived())
                {
                    if (useNAVIAIWholeRobotSendCtrl_)
                    {
                        teleoperationSetTargetPublishers_[1].publish(finalTarget_);
                        auto copiedTarget = finalTarget_;
                        copiedTarget.leftArmValid = copiedTarget.rightArmValid = copiedTarget.leftHandValid = copiedTarget.rightHandValid =
                            copiedTarget.waistValid = false;
                        teleoperationSetTargetPublishers_[0].publish(copiedTarget);
                    }
                    else
                        teleoperationSetTargetPublishers_[0].publish(finalTarget_);
                    waistHaveControlledFlag_ = true;
                }
            }
            catch (const std::exception &e)
            {
                ROS_ERROR("Get error %s while publishing target angles for robot body.", e.what());
            }
        }
    }

    bool checkWaistErrorValid_(const float *target)
    {
        if (!waistHaveControlledFlag_)
        {
            const robot_uplimb_pkg::WholeBodyPositionVelocity &jointState = getNewJointStates();
            for (size_t i = 0; i < lastWaistTarget_.size(); ++i)
                lastWaistTarget_.at(i) = jointState.waistPosition.at(i);
            for (size_t i = 0; i < lastWaistTarget_.size(); ++i)
                if (std::abs(lastWaistTarget_.at(i) - target[i]) > 0.3)
                    return false;
        }
        else
            for (size_t i = 0; i < lastWaistTarget_.size(); ++i)
                if (std::abs(lastWaistTarget_.at(i) - target[i]) > 0.3)
                    return false;
        return true;
    }

    void moveWaistFirstPoint_(void)
    {
        if (waistMoveJMoving_.load())
            return;
#if (defined USING_NAVIAI_ROBOT_WA1_0303) || (defined USING_NAVIAI_ROBOT_WA2_A2_LITE)
        waistMoveJMoving_ = true;
        teleoperationControlWheeledWaistFromVRClient_.call(teleoperationControlWheeledWaistSRV_);
        for (size_t i = 0; i < teleoperationControlWheeledWaistSRV_.request.jnt_angle.size(); ++i)
            lastWaistTarget_.at(i) = teleoperationControlWheeledWaistSRV_.request.jnt_angle.at(i);
#endif
        waistMoveJMoving_ = false;
    }

    void sendHandTarget_(bool isLeft)
    {
#if (TEST_MODE == 0) || (TEST_DIFFUSION == 1)
        auto &finalTarget = isLeft ? finalTarget_.target.leftHandPosition : finalTarget_.target.rightHandPosition;
        auto &finalTargetValid = isLeft ? finalTarget_.leftHandValid : finalTarget_.rightHandValid;
        if (finalTargetValid)
        {
            if (!controlHandServiceClient_.exists())
            {
                ROS_WARN("Hand service client is not available.");
                return;
            }
            navi_types::RobotHandJointSrv srv;
            srv.request.id = isLeft ? 0 : 1;
            srv.request.q.resize(6);
            for (size_t i = 0; i < 6; ++i)
                srv.request.q[i] = finalTarget[i];
            if (controlHandServiceClient_.call(srv))
            {
                if (!srv.response.success)
                    ROS_WARN("Failed to control %s hand: %s", isLeft ? "Left" : "Right", srv.response.message.c_str());
            }
            else
                ROS_ERROR("Failed to call service: /robotHandJointSwitch");
        }
#endif
    }

    void controlRobotHandThread_(void)
    {
        std::this_thread::sleep_for(std::chrono::seconds(3));
        ros::Rate runRate(stepFrequency_);
        while (isRunning() && ros::ok())
        {
            runRate.sleep();
            if (!getMovePermission())
                continue;
            try
            {
                sendHandTarget_(true);
                sendHandTarget_(false);
            }
            catch (const std::exception &e)
            {
                ROS_ERROR("Get error %s while publishing target angles for robot.", e.what());
            }
        }
    }

#if (defined USING_NAVIAI_ROBOT_WA1_0303) || (defined USING_NAVIAI_ROBOT_WA2_A2_LITE)
    void controlChassisThread_(void)
    {
        geometry_msgs::Twist zeroCmd;
        if (controlWheeledRobotChassisWaistUsingController_)
        {
            std::this_thread::sleep_for(std::chrono::seconds(3));
            ros::Rate runRate(60);
            while (isRunning() && ros::ok())
            {
                runRate.sleep();
                try
                {
                    if (getUnixTimestampInSeconds() - lastRecvTargetTick_ > 1.5)
                        teleoperationControlChassisPublisher_.publish(zeroCmd);
                    else
                        teleoperationControlChassisPublisher_.publish(chassisCmd_);
                }
                catch (const std::exception &e)
                {
                    ROS_ERROR("Get error %s while publishing target for chassis.", e.what());
                }
            }
        }
    }

    void controlWheeledWaistFromVRThread_(void)
    {
#if defined CALL_WHEELED_WAIST_MOVE_SERVICE
        if (controlWheeledRobotChassisWaistUsingController_)
        {
            std::this_thread::sleep_for(std::chrono::seconds(3));
            ros::Rate runRate(60);
            while (isRunning() && ros::ok())
            {
                runRate.sleep();
                try
                {
                    if (getUnixTimestampInSeconds() - lastRecvTargetTick_ < 1.5)
                    {
                        if (!teleoperationControlWheeledWaistFromVRClient_.exists())
                        {
                            ROS_WARN("Waist MoveJ service client is not available.");
                            return;
                        }
                        teleoperationControlWheeledWaistSRV_.request.not_wait = false;
                        // if (teleoperationControlWheeledWaistSRV_.request.jnt_angle.size() == 2)
                        //     ROS_INFO("Waist target: [%lf, %lf]", teleoperationControlWheeledWaistSRV_.request.jnt_angle.at(0),
                        //              teleoperationControlWheeledWaistSRV_.request.jnt_angle.at(1));
                        // else
                        //     ROS_INFO("Waist target: %lf", teleoperationControlWheeledWaistSRV_.request.jnt_angle.at(0));
                        // if (teleoperationControlWheeledWaistFromVRClient_.call(teleoperationControlWheeledWaistSRV_))
                        // {
                        //     if (!teleoperationControlWheeledWaistSRV_.response.finish.data)
                        //         ROS_WARN("Failed to control wheeled waist.");
                        // }
                        // else
                        //     ROS_ERROR("Failed to call service: /waist_movej_service");
                        if (!teleoperationControlWheeledWaistFromVRClient_.call(teleoperationControlWheeledWaistSRV_))
                            ROS_ERROR("Failed to call service: /waist_movej_service");
                    }
                }
                catch (const std::exception &e)
                {
                    ROS_ERROR("Get error %s while calling waist control service.", e.what());
                }
            }
        }
#endif
    }
#endif

    void recordRosbagThread_(void)
    {
#if (TEST_MODE == 0)
        std::this_thread::sleep_for(std::chrono::seconds(2));
        if (!isRunning())
            return;
        ROS_INFO("Record ROSBAG thread start.");
        std::string shellName = packagePath_ + "/orin_record_topics.sh";
        std::unique_ptr<boost::process::child> rosbagProcess;
        std_msgs::Bool recordStateMsg;
        auto callRecordService = [&](bool start) {
            if (!recordServiceReadyFlag_)
                ROS_ERROR("/hdf5/start_recording and /hdf5/stop_recording service not ready, could not control record");
            using namespace std::chrono_literals;
            if (!recordServiceMutex_.try_lock_for(2s))
            {
                ROS_WARN("Failed to acquire lock for /hdf5/%s_recording within 2 seconds, skipping call.", start ? "start" : "stop");
                return;
            }
            std::lock_guard<std::timed_mutex> lock(recordServiceMutex_, std::adopt_lock);
            try
            {
                std_srvs::Empty srv;
                if (recordServiceClient_.at(start ? 0 : 1).call(srv))
                    ROS_INFO("Successfully called /hdf5/%s_recording service.", start ? "start" : "stop");
                else
                    ROS_ERROR("Failed to call /hdf5/%s_recording service.", start ? "start" : "stop");
            }
            catch (const std::exception &e)
            {
                ROS_ERROR("Exception while calling /hdf5/%s_recording: %s", start ? "start" : "stop", e.what());
            }
            catch (...)
            {
                ROS_ERROR("Unknown exception in /hdf5/%s_recording service call.", start ? "start" : "stop");
            }
        };
        auto stopRecord = [&shellName, callRecordService](std::unique_ptr<boost::process::child> &rosbagProcess) -> void {
            try
            {
                std::thread([&]() { callRecordService(false); }).detach();
                if (rosbagProcess && rosbagProcess->running())
                {
                    rosbagProcess->terminate();
                    rosbagProcess->wait();
                    rosbagProcess.reset();
                }
                std::string stop_shell = shellName + " --clean";
                int res = std::system(stop_shell.c_str());
                ROS_INFO("Run clean code with return value %d", res);
            }
            catch (const std::exception &e)
            {
                ROS_ERROR_STREAM("Error stopping rosbag process: " << e.what());
            }
        };
        auto startRecord = [&shellName, stopRecord, callRecordService](std::unique_ptr<boost::process::child> &rosbagProcess) -> void {
            try
            {
                stopRecord(rosbagProcess);
                std::string start_shell = shellName + " --record";
                std::thread([&]() { callRecordService(true); }).detach();
                rosbagProcess = std::make_unique<boost::process::child>(boost::process::search_path("bash"), "-c", start_shell);
                ROS_INFO_STREAM("Started rosbag recording with PID: " << rosbagProcess->id());
            }
            catch (const std::exception &e)
            {
                ROS_ERROR_STREAM("Error starting rosbag process: " << e.what());
            }
        };
        while (ros::ok() && isRunning())
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(30));
            bool recordDataFlag = recordDataFlag_.load();
            if (recordDataFlag && !rosbagProcess)
                startRecord(rosbagProcess);
            else if (!recordDataFlag && rosbagProcess && rosbagProcess->running())
                stopRecord(rosbagProcess);
            recordStateMsg.data = recordDataFlag;
            teleoperationRecordDataStatePublisher_.publish(recordStateMsg);
        }
        stop();
        stopRecord(rosbagProcess);
#endif
    }
};

int main(int argc, char **argv)
{
    ros::init(argc, argv, NODE_NAME);
    Naviai_UDP_Communicate_Robot communicationHandler(false);
    communicationHandler.start();
    return 0;
}
