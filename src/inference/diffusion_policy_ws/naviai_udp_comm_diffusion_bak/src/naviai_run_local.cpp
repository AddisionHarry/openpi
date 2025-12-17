#include "app/naviai_app.h"
#include "app/robot_control.h"
#include "utils.h"

#include <boost/process.hpp>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <stdexcept>
#include <thread>

#include <ros/package.h>
#include <ros/ros.h>

#include "navi_types/RobotHandJointSrv.h"
#include "naviai_udp_comm/TeleoperationUDPRaw.h"
#include "robot_uplimb_pkg/TeleoperationTarget.h"
#include "system_state_msgs/system_state_update.h"
#include <sensor_msgs/JointState.h>
#include <std_msgs/Bool.h>
#include <std_msgs/Float32MultiArray.h>
#include <std_msgs/Float64.h>
#include <std_msgs/Float64MultiArray.h>

#define NODE_NAME "teleoperation_local_control_node"

#define QOS_FREQ_FILTER_RATE 0.02
#define QOS_FREQ_THRESHOLD 0.4

class Naviai_Control_Robot : private Naviai_UDP_Communicate_App, public Robot_Control_Target
{
  public:
    Naviai_Control_Robot()
        : Naviai_UDP_Communicate_App(NODE_NAME, 0, 0, 0, false), Robot_Control_Target(true, true, false), recordDataFlag_(false),
          firstRecvFlag_(true)
    {
        initROSNode_();
        loadConfig_();
    }

    ~Naviai_Control_Robot()
    {
        Naviai_UDP_Communicate_App::stop();
    }

    void start(void)
    {
        std::thread(&Naviai_Control_Robot::controlRobotBodyThread_, this).detach();
        std::thread(&Naviai_Control_Robot::recordRosbagThread_, this).detach();
        std::thread(&Naviai_Control_Robot::sendCommunicationStateThread_, this).detach();
        Naviai_UDP_Communicate_App::start();
        ros::spin();
    }

  private:
    double stepFrequency_, cmdFrequency_;
    std::string packagePath_;
    HandleControlFlags serverSideControlFlags_;

    volatile bool recordDataFlag_;
    std::array<double, BODY_JOINT_NUM> recvJointPositions_;
    std::array<double, BODY_JOINT_NUM> recvJointVeclocities_;

    ros::Publisher teleoperationSetTargetPublisher_;
    ros::Publisher teleoperationRecordDataStatePublisher_;
    ros::Publisher systemStateUpdatePublisher_;
    ros::Publisher jointStatesFeedbackPublisher_;
    ros::Publisher jointStatesLatestFeedbackTimePublishers_;
    ros::Subscriber jointStatesSubscriber_;
    ros::Subscriber contrTargetSubscriber_;
    ros::Subscriber recordMsgSubscriber_;

    naviai_udp_comm::TeleoperationUDPRaw recvRawTarget_;
    robot_uplimb_pkg::TeleoperationTarget finalTarget_;

    bool firstRecvFlag_;
    double realCalculatedFPS_;
    double lastRecvTick_;
    bool communicationValidFlag_;

    void initROSNode_(void) override
    {
        stepFrequency_ = getParam_<double>("/step_frequency", 300.0);
        cmdFrequency_ = getParam_<double>("/command_frequency", 300.0);
        realCalculatedFPS_ = cmdFrequency_;
        setPlanTime(1.0 / stepFrequency_ * 1.02);
        packagePath_ = ros::package::getPath("naviai_udp_comm");
        if (packagePath_.empty())
        {
            ROS_ERROR("Failed to find package naviai_udp_comm.");
            return;
        }
        teleoperationSetTargetPublisher_ = nh.advertise<robot_uplimb_pkg::TeleoperationTarget>("/teleoperation_ctrl_cmd/final", 10);
        teleoperationRecordDataStatePublisher_ = nh.advertise<std_msgs::Bool>("/teleoperation_ctrl_cmd/record_data", 10);
        systemStateUpdatePublisher_ = nh.advertise<system_state_msgs::system_state_update>("/system_state/update_topic", 20);
        jointStatesFeedbackPublisher_ = nh.advertise<sensor_msgs::JointState>("/joint_states_feedback/raw", 10);
        jointStatesLatestFeedbackTimePublishers_ = nh.advertise<std_msgs::Float64>("/joint_states_feedback/lastest_feedback_time", 10);
        jointStatesSubscriber_ = nh.subscribe("/joint_states", 10, &Naviai_Control_Robot::onJointStatesReceive_, this);
        contrTargetSubscriber_ = nh.subscribe("/teleoperation_target/final", 10, &Naviai_Control_Robot::onTargetsReceive_, this);
        recordMsgSubscriber_ = nh.subscribe("/system_state/record_state", 10, &Naviai_Control_Robot::onRecordCmdReceive_, this);
    }

    void loadConfig_(void)
    {
        if (packagePath_.empty())
        {
            ROS_ERROR("Failed to find package naviai_udp_comm.");
            return;
        }
        std::string configPath = packagePath_ + "/config/server.yaml";
        auto config = loadRobotConfig(configPath);
        serverSideControlFlags_.setControlFlag(Robot_Control_Type::Neck_Target,
                                               config["Control_Tasks"]["Control_Neck"].as<std::string>() == "true");
        serverSideControlFlags_.setControlFlag(Robot_Control_Type::Left_Arm_Target,
                                               config["Control_Tasks"]["Control_Arm"].as<std::string>() == "true");
        serverSideControlFlags_.setControlFlag(Robot_Control_Type::Left_Hand_Target,
                                               config["Control_Tasks"]["Control_Hand"].as<std::string>() == "true");
        serverSideControlFlags_.setControlFlag(Robot_Control_Type::Waist_Target,
                                               config["Control_Tasks"]["Control_Waist"].as<std::string>() == "true");
        printConfig(config);
    }

    void onUDPDataRecv_(std::vector<std::uint8_t> &&data, NAVIAI_UDP_Communication::Packet_Type_Enum type, std::uint64_t sendTimestamp,
                        std::uint64_t recvTimestamp, const std::string &sender_ip, int sender_port) override
    {
        return;
    }

    void onRecordCmdReceive_(const std_msgs::BoolConstPtr &msgptr)
    {
        if (!isRunning())
            return;
        recordDataFlag_ = msgptr->data;
    }

    void updateRecvTargetFreq_(void)
    {
        if (!firstRecvFlag_)
            realCalculatedFPS_ = realCalculatedFPS_ * (1 - QOS_FREQ_FILTER_RATE) +
                                 1.0 / (getUnixTimestampInSeconds() - lastRecvTick_) * QOS_FREQ_FILTER_RATE;
        lastRecvTick_ = getUnixTimestampInSeconds();
        firstRecvFlag_ = false;
        if (realCalculatedFPS_ < cmdFrequency_ * QOS_FREQ_THRESHOLD)
        {
            ROS_ERROR("Communication quality bad! Expected %lf fps, get %lf fps.", cmdFrequency_, realCalculatedFPS_);
            communicationValidFlag_ = false;
        }
        else
            communicationValidFlag_ = true;
    }

    void sendCommunicationStateThread_(void)
    {
        std::this_thread::sleep_for(std::chrono::seconds(10));
        if (!isRunning())
            return;
        system_state_msgs::system_state_update msg;
        ros::Rate rate(60);
        for (size_t i = 0; i < msg.data.size(); ++i)
            msg.valid[i] = 0;
        while (ros::ok() && isRunning())
        {
            rate.sleep();
            msg.valid[1] = true;
            msg.data[1] = static_cast<std::uint8_t>(communicationValidFlag_);
            msg.timestamp = ros::Time::now();
            systemStateUpdatePublisher_.publish(msg);
        }
    }

    void onTargetsReceive_(const naviai_udp_comm::TeleoperationFinalTargetConstPtr &msgptr)
    {
        if (!isRunning())
            return;
        if ((!getNumberIsNormal(msgptr->waistTargetForWheeled)) || (!getNumberIsNormal(msgptr->chassisCmd)) ||
            (!getNumberIsNormal(msgptr->target)))
        {
            ROS_ERROR("Get nan/inf in target angles.");
            return;
        }
        recvRawTarget_.recordingFlag = recordDataFlag_;
        std::uint8_t validFlag = 0x00;
        for (size_t i = 0; i < static_cast<size_t>(Robot_Control_Type::Target_All); ++i)
        {
            Robot_Control_Type type = static_cast<Robot_Control_Type>(i);
            serverSideControlFlags_.setControlValidFlag(validFlag, type, serverSideControlFlags_.getControlFlag(type));
        }
        recvRawTarget_.controlFlag = validFlag;
        recvRawTarget_.target = *msgptr;
        setNewTarget(boost::make_shared<robot_uplimb_pkg::WholeBodyPositionVelocity>(recvRawTarget_.target.target),
                     recvRawTarget_.controlFlag);
    }

    void onJointStatesReceive_(const sensor_msgs::JointStateConstPtr &msgptr)
    {
        try
        {
            if (!isRunning())
                return;
            updateRecvTargetFreq_();
            publishNewJointStateFeedback_(msgptr);
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
                    continue;
                if ((i == msg.name.size() - 1) || (foundJointCNT == BODY_JOINT_NUM))
                    updateNewJointState(recvJointPositions_, recvJointVeclocities_);
            }
        }
        catch (const std::exception &e)
        {
            ROS_ERROR("Get error %s while receiving /joint_states topic.", e.what());
        }
    }

    void publishNewJointStateFeedback_(const sensor_msgs::JointStateConstPtr &msgptr)
    {
        std_msgs::Float64 tick_msg;
        tick_msg.data = getUnixTimestampInSeconds();
        jointStatesFeedbackPublisher_.publish(*msgptr);
        jointStatesLatestFeedbackTimePublishers_.publish(tick_msg);
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
                updateNewTarget();
                finalTarget_.neckValid = getTargetValidation(Robot_Control_Type::Neck_Target);
                finalTarget_.leftArmValid = getTargetValidation(Robot_Control_Type::Left_Arm_Target);
                finalTarget_.leftHandValid = getTargetValidation(Robot_Control_Type::Left_Hand_Target);
                finalTarget_.rightArmValid = getTargetValidation(Robot_Control_Type::Right_Arm_Target);
                finalTarget_.rightHandValid = getTargetValidation(Robot_Control_Type::Right_Hand_Target);
                finalTarget_.waistValid = getTargetValidation(Robot_Control_Type::Waist_Target);
                finalTarget_.leftArmSlowMoveJFlag = getSlowMoveJFlag()[0];
                finalTarget_.rightArmSlowMoveJFlag = getSlowMoveJFlag()[1];
                finalTarget_.target = getCurrentNewTarget();
                teleoperationSetTargetPublisher_.publish(finalTarget_);
            }
            catch (const std::exception &e)
            {
                ROS_ERROR("Get error %s while publishing target angles for robot body.", e.what());
            }
        }
    }

    void recordRosbagThread_(void)
    {
        std::this_thread::sleep_for(std::chrono::seconds(2));
        if (!isRunning())
            return;
        std::string shellName = packagePath_ + "/orin_record_topics.sh";
        std::unique_ptr<boost::process::child> rosbagProcess;
        std_msgs::Bool recordStateMsg;
        auto stopRecord = [&shellName](std::unique_ptr<boost::process::child> &rosbagProcess) -> void {
            try
            {
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
        auto startRecord = [&shellName, stopRecord](std::unique_ptr<boost::process::child> &rosbagProcess) -> void {
            try
            {
                stopRecord(rosbagProcess);
                std::string start_shell = shellName + " --record";
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
            if (recordDataFlag_ && !rosbagProcess)
                startRecord(rosbagProcess);
            else if (!recordDataFlag_ && rosbagProcess && rosbagProcess->running())
                stopRecord(rosbagProcess);
            recordStateMsg.data = recordDataFlag_;
            teleoperationRecordDataStatePublisher_.publish(recordStateMsg);
        }
        stop();
        stopRecord(rosbagProcess);
    }
};

int main(int argc, char **argv)
{
    ros::init(argc, argv, NODE_NAME);
    Naviai_Control_Robot controlHandler;
    controlHandler.start();
    return 0;
}
