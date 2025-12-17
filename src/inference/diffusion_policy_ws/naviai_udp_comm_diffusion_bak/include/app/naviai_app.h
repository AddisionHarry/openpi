#ifndef NAVIAI_APP_H
#define NAVIAI_APP_H

#include "communication/naviai_udp_protocal.h"
#include "utils.h"

#include "ros/ros.h"
#include <array>
#include <unordered_map>
#include <vector>

#include <ros/serialization.h>

#include "robot_uplimb_pkg/WholeBodyPositionVelocity.h"
#include <geometry_msgs/Twist.h>
#include <geometry_msgs/Vector3.h>

class Naviai_UDP_Communicate_App
{
  public:
    ros::NodeHandle nh;

    std::unordered_map<std::string, size_t> jointName2IndexMap;

    Naviai_UDP_Communicate_App(std::string nodeName, int commonSendPort, int videoPort, int recvPort, bool needUDP);
    virtual ~Naviai_UDP_Communicate_App();

    std::vector<ssize_t> packSendData(const std::uint8_t *data, size_t length, NAVIAI_UDP_Communication::Packet_Type_Enum type) const;
    std::vector<ssize_t> packSendData(const std::vector<std::uint8_t> &data, NAVIAI_UDP_Communication::Packet_Type_Enum type) const;

    void start(void);
    void stop(void);
    bool isRunning(void);

  protected:
    std::string nodeName_;
    std::string targetIP_;
    int commonSendPort_;
    int videoPort_;
    int recvPort_;
    bool runFlag_;
    bool needUDP_;

    template <typename T> T getParam_(const std::string &paramNameLocal, const T &defaultValue) const
    {
        std::string paramName = (nodeName_[0] == '/') ? (nodeName_ + paramNameLocal) : ("/" + nodeName_ + paramNameLocal);
        return ros::param::param<T>(paramName, static_cast<T>(defaultValue));
    }

    void loadConfigFromFile_(std::ifstream &file);
    void initUDPCommunication_(void);
    void onUDPRecv_(const std::uint8_t *data, size_t dataLength, const std::string &senderIP, int senderPort);

    virtual void initROSNode_(void) = 0;
    virtual void onUDPDataRecv_(std::vector<std::uint8_t> &&data, NAVIAI_UDP_Communication::Packet_Type_Enum type,
                                std::uint64_t sendTimestamp, std::uint64_t recvTimestamp, const std::string &senderIP, int senderPort) = 0;
};

extern bool getNumberIsNormal(const geometry_msgs::Vector3 &v);
extern bool getNumberIsNormal(const geometry_msgs::Twist &w);
extern bool getNumberIsNormal(const robot_uplimb_pkg::WholeBodyPositionVelocity &w);

template <typename MsgType> std::vector<std::uint8_t> serializeRosMessage(const MsgType &msg)
{
    namespace ser = ros::serialization;
    std::uint32_t serial_size = ser::serializationLength(msg);
    std::vector<std::uint8_t> buffer(serial_size);
    ser::OStream stream(buffer.data(), buffer.size());
    ser::Serializer<MsgType>::write(stream, msg);
    return buffer;
}

template <typename MsgType> bool deserializeRosMessage(const std::vector<std::uint8_t> &buffer, MsgType &msg)
{
    namespace ser = ros::serialization;
    if (buffer.empty())
        return false;
    ser::IStream stream(const_cast<uint8_t *>(buffer.data()), buffer.size());
    try
    {
        ser::Serializer<MsgType>::read(stream, msg);
    }
    catch (const std::exception &e)
    {
        ROS_ERROR("Deserialization error: %s", e.what());
        return false;
    }
    return true;
}

#endif /* NAVIAI_APP_H */
