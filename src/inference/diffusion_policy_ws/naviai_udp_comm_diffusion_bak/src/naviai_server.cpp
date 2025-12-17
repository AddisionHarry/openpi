#include "app/naviai_app.h"
#include "communication/naviai_udp_protocal.h"
#include "naviai_udp_comm/TeleoperationFinalTarget.h"
#include "naviai_udp_comm/TeleoperationUDPRaw.h"
#include "ros/console.h"
#include "ros/rate.h"
#include "utils.h"

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <thread>

#include <ros/package.h>
#include <ros/ros.h>

#include "system_state_msgs/system_state_update.h"

#include <sensor_msgs/JointState.h>
#include <std_msgs/Bool.h>
#include <std_msgs/Float64.h>
#include <std_msgs/Float64MultiArray.h>

#define NODE_NAME "teleoperation_communication_server_node"

#define QOS_FREQ_FILTER_RATE 0.03
#define QOS_FREQ_THRESHOLD 0.6

typedef union
{
    float data;
    std::uint8_t bytes[4];
} CvtFloatBytes_u;

class Naviai_UDP_Communicate_Server : private Naviai_UDP_Communicate_App
{
  public:
    Naviai_UDP_Communicate_Server()
        : Naviai_UDP_Communicate_App(NODE_NAME, 28256, 28258, 28257, true), recordState_(false), firstRecvFlag_(true),
          lastRecvTick_(getUnixTimestampInSeconds()), communicationValidFlag_(false)
    {
        loadConfig_();
        initROSNode_();
    }

    ~Naviai_UDP_Communicate_Server()
    {
        Naviai_UDP_Communicate_App::stop();
    }

    void start(void)
    {
        lastRecvTick_ = getUnixTimestampInSeconds();
        // std::thread(&Naviai_UDP_Communicate_Server::printBufferState_, this).detach();
        std::thread(&Naviai_UDP_Communicate_Server::sendCommunicationStateThread_, this).detach();
        Naviai_UDP_Communicate_App::start();
        ros::spin();
    }

  private:
    ros::Publisher forceFeedbackPublisher_;
    ros::Publisher jointStatesFeedbackPublisher_;
    ros::Publisher jointStatesLatestFeedbackTimePublisher_;
    ros::Publisher systemStateUpdatePublisher_;
    ros::Subscriber controlTargetSubscriber_;
    ros::Subscriber recordMsgSubscriber_;
    HandleControlFlags flags_;
    naviai_udp_comm::TeleoperationUDPRaw udpSendData_;
    bool recordState_;

    sensor_msgs::JointState feedbackJointStates_;
    std_msgs::Float64MultiArray feedbackForce_;

    double stepFrequency_, cmdFrequency_;
    bool firstRecvFlag_;
    double realCalculatedFPS_;
    double lastRecvTick_;
    bool communicationValidFlag_;

    void printBufferState_(void)
    {
        std::this_thread::sleep_for(std::chrono::seconds(1));
        ros::Rate loop_rate(0.3);
        while (ros::ok())
        {
            loop_rate.sleep();
            if ((NAVIAI_UDP_Communication::getCurrentBufferUsedSize() > 100U) ||
                (NAVIAI_UDP_Communication::getMultiFrameBufferUsedSize() > 100U) ||
                (NAVIAI_UDP_Communication::getCurrentBufferOverflowCNT() > 10U))
                ROS_ERROR(
                    "Current using buffer size: %ld bytes / %ld bytes, multipack buffer size: %ld bytes, overflow flag set %ld times.",
                    NAVIAI_UDP_Communication::getCurrentBufferUsedSize(), NAVIAI_UDP_Communication::getCurrentBufferMaxSize(),
                    NAVIAI_UDP_Communication::getMultiFrameBufferUsedSize(), NAVIAI_UDP_Communication::getCurrentBufferOverflowCNT());
        }
        stop();
    }

    void loadConfig_(void)
    {
        std::string packagePath = ros::package::getPath("naviai_udp_comm");
        if (packagePath.empty())
        {
            ROS_ERROR("Failed to find package naviai_udp_comm.");
            return;
        }
        std::string configPath = packagePath + "/config/server.yaml";
        auto config = loadRobotConfig(configPath);
        flags_.setControlFlag(Robot_Control_Type::Neck_Target, config["Control_Tasks"]["Control_Neck"].as<std::string>() == "true");
        flags_.setControlFlag(Robot_Control_Type::Left_Arm_Target, config["Control_Tasks"]["Control_Arm"].as<std::string>() == "true");
        flags_.setControlFlag(Robot_Control_Type::Left_Hand_Target, config["Control_Tasks"]["Control_Hand"].as<std::string>() == "true");
        flags_.setControlFlag(Robot_Control_Type::Waist_Target, config["Control_Tasks"]["Control_Waist"].as<std::string>() == "true");
        printConfig(config);
    }

    void initROSNode_(void) override
    {
        stepFrequency_ = getParam_<double>("/step_frequency", 300.0);
        cmdFrequency_ = getParam_<double>("/command_frequency", 300.0);
        realCalculatedFPS_ = cmdFrequency_;
        feedbackJointStates_.name = std::vector<std::string>(jointNames.begin(), jointNames.end());
        feedbackJointStates_.position = std::vector<double>(feedbackJointStates_.name.size(), 0.0);
        feedbackJointStates_.velocity = std::vector<double>(feedbackJointStates_.name.size(), 0.0);
        jointStatesFeedbackPublisher_ = nh.advertise<sensor_msgs::JointState>("/joint_states_feedback/raw", 10);
        systemStateUpdatePublisher_ = nh.advertise<system_state_msgs::system_state_update>("/system_state/update_topic", 20);
        jointStatesLatestFeedbackTimePublisher_ = nh.advertise<std_msgs::Float64>("/joint_states_feedback/lastest_feedback_time", 10);
        forceFeedbackPublisher_ = nh.advertise<std_msgs::Float64MultiArray>("/force_feedback", 10);
        controlTargetSubscriber_ = nh.subscribe("/teleoperation_target/final", 10, &Naviai_UDP_Communicate_Server::onTargetsReceive_, this);
        recordMsgSubscriber_ = nh.subscribe("/system_state/record_state", 10, &Naviai_UDP_Communicate_Server::onRecordCmdReceive_, this);
        feedbackForce_.data.resize(12);
    }

    void onRecordCmdReceive_(const std_msgs::BoolConstPtr &msgptr)
    {
        if (!isRunning())
            return;
        recordState_ = msgptr->data;
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
        udpSendData_.recordingFlag = recordState_;
        std::uint8_t validFlag = 0x00;
        for (size_t i = 0; i < static_cast<size_t>(Robot_Control_Type::Target_All); ++i)
        {
            Robot_Control_Type type = static_cast<Robot_Control_Type>(i);
            flags_.setControlValidFlag(validFlag, type, flags_.getControlFlag(type));
        }
        udpSendData_.controlFlag = validFlag;
        udpSendData_.target = *msgptr;
        auto res = packSendData(serializeRosMessage(udpSendData_), NAVIAI_UDP_Communication::Packet_Type_Enum::Packed_Target);
        for (auto r : res)
            if (r <= 0)
                ROS_ERROR("Send joints target data failed.");
    }

    void updateRecvTargetFreq_(void)
    {
        if ((!firstRecvFlag_) && (getUnixTimestampInSeconds() - lastRecvTick_ > 1e-4))
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

    void onUDPDataRecv_(std::vector<std::uint8_t> &&data, NAVIAI_UDP_Communication::Packet_Type_Enum type, std::uint64_t sendTimestamp,
                        std::uint64_t recvTimestamp, const std::string &sender_ip, int sender_port) override
    {
        if (!isRunning())
            return;
        switch (type)
        {
        case NAVIAI_UDP_Communication::Packet_Type_Enum::Test_Latency:
            break;
        case NAVIAI_UDP_Communication::Packet_Type_Enum::Joint_States_Feedback:
            updateRecvTargetFreq_();
            onJointStatesFeedbackRecv_(std::move(data));
            break;
        case NAVIAI_UDP_Communication::Packet_Type_Enum::Force_Feedback:
            onForceFeedbackRecv_(std::move(data));
            break;
        case NAVIAI_UDP_Communication::Packet_Type_Enum::Packed_Target:
            ROS_ERROR("Get invalid feedback type of packed target.");
            break;
        default:
            ROS_ERROR("Get unchecked feedback type: %d", static_cast<int>(type));
            break;
        }
    }

    void onForceFeedbackRecv_(std::vector<std::uint8_t> &&data)
    {
        if (data.size() != 12 * 4)
        {
            ROS_WARN("Get invalid force feedback data length: %ld.", data.size());
            return;
        }
        CvtFloatBytes_u *cvt = nullptr;
        for (size_t i = 0, j = 0; i < data.size(); i += 4, ++j)
        {
            CvtFloatBytes_u *cvt = reinterpret_cast<CvtFloatBytes_u *>(&data[i]);
            feedbackForce_.data.at(j) = static_cast<double>(cvt->data);
        }
        forceFeedbackPublisher_.publish(feedbackForce_);
    }

    void onJointStatesFeedbackRecv_(std::vector<std::uint8_t> &&data)
    {
        robot_uplimb_pkg::WholeBodyPositionVelocity unpackedJointStates;
        if (!deserializeRosMessage(data, unpackedJointStates))
        {
            ROS_ERROR("Get error while unpacking joint states feedback.");
            return;
        }
        std_msgs::Float64 tickMsg;
        tickMsg.data = getUnixTimestampInSeconds();
        CvtFloatBytes_u *cvt = nullptr;
        feedbackJointStates_.header.stamp = ros::Time::now();
        for (size_t i = 0; i < LEFT_ARM_JOINT_NUM; ++i)
        {
            feedbackJointStates_.position.at(i + RIGHT_ARM_INDEX[0]) = unpackedJointStates.rightArmPosition.at(i);
            feedbackJointStates_.velocity.at(i + RIGHT_ARM_INDEX[0]) = unpackedJointStates.rightArmVelocity.at(i);
            feedbackJointStates_.position.at(i + LEFT_ARM_INDEX[0]) = unpackedJointStates.leftArmPosition.at(i);
            feedbackJointStates_.velocity.at(i + LEFT_ARM_INDEX[0]) = unpackedJointStates.leftArmVelocity.at(i);
        }
        for (size_t i = 0; i < NECK_JOINT_NUM; ++i)
        {
            feedbackJointStates_.position.at(i + NECK_INDEX[0]) = unpackedJointStates.neckPosition.at(i);
            feedbackJointStates_.velocity.at(i + NECK_INDEX[0]) = unpackedJointStates.neckVelocity.at(i);
        }
        for (size_t i = 0; i < WAIST_JOINT_NUM; ++i)
        {
            feedbackJointStates_.position.at(i + WAIST_INDEX[0]) = unpackedJointStates.waistPosition.at(i);
            feedbackJointStates_.velocity.at(i + WAIST_INDEX[0]) = unpackedJointStates.waistVelocity.at(i);
        }
        jointStatesFeedbackPublisher_.publish(feedbackJointStates_);
        jointStatesLatestFeedbackTimePublisher_.publish(tickMsg);
    }
};

int main(int argc, char **argv)
{
    ros::init(argc, argv, NODE_NAME);
    Naviai_UDP_Communicate_Server comm;
    comm.start();
    return 0;
}
