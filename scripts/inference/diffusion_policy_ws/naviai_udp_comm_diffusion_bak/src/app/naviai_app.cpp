#include "app/naviai_app.h"
#include "communication/udp_comm.h"
#include "utils.h"

#include <cstddef>
#include <fstream>
#include <future>
#include <sstream>

Naviai_UDP_Communicate_App::Naviai_UDP_Communicate_App(std::string nodeName, int commonSendPort, int videoPort, int recvPort, bool needUDP)
    : nodeName_(nodeName), commonSendPort_(commonSendPort), videoPort_(videoPort), recvPort_(recvPort), runFlag_(false), needUDP_(needUDP)
{
    for (size_t i = 0; i < jointNames.size(); ++i)
        jointName2IndexMap[jointNames[i]] = i;

    if (needUDP_)
        initUDPCommunication_();
}

Naviai_UDP_Communicate_App::~Naviai_UDP_Communicate_App()
{
    runFlag_ = false;
    using namespace NAVIAI_UDP_Communication;
    udp_sender_shutdown();
    udp_receive_stop_all_receivers();
}

std::vector<ssize_t> Naviai_UDP_Communicate_App::packSendData(const std::uint8_t *data, size_t length,
                                                              NAVIAI_UDP_Communication::Packet_Type_Enum type) const
{
    using namespace NAVIAI_UDP_Communication;
    auto packed = packData(data, length, type);
    std::vector<ssize_t> res(packed.size());
    for (size_t i = 0; i < packed.size(); ++i)
        res.at(i) = udp_sender_send(reinterpret_cast<const std::uint8_t *>(packed.at(i).data()), packed.at(i).size());
    return res;
}

std::vector<ssize_t> Naviai_UDP_Communicate_App::packSendData(const std::vector<std::uint8_t> &data,
                                                              NAVIAI_UDP_Communication::Packet_Type_Enum type) const
{
    return packSendData(reinterpret_cast<const std::uint8_t *>(data.data()), data.size(), type);
}

void Naviai_UDP_Communicate_App::start(void)
{
    runFlag_ = true;
}

void Naviai_UDP_Communicate_App::stop(void)
{
    runFlag_ = false;
}

bool Naviai_UDP_Communicate_App::isRunning(void)
{
    return runFlag_;
}

void Naviai_UDP_Communicate_App::initUDPCommunication_(void)
{
    std::string paramName = (nodeName_[0] == '/') ? (nodeName_ + "/target_ip") : ("/" + nodeName_ + "/target_ip");
    targetIP_ = ros::param::param<std::string>(paramName, "");
    ROS_INFO("Get udp target IP: %s.", targetIP_.c_str());

    using namespace NAVIAI_UDP_Communication;
    if (!init_udp_sender(targetIP_, commonSendPort_))
    {
        ROS_ERROR("Start sending on port %d failed!", commonSendPort_);
        return;
    }
    if (!init_udp_receiver(recvPort_))
    {

        ROS_ERROR("Start receiving on port %d failed!", recvPort_);
        return;
    }

    udp_receive_register_callback(recvPort_, std::bind(&Naviai_UDP_Communicate_App::onUDPRecv_, this, std::placeholders::_1,
                                                       std::placeholders::_2, std::placeholders::_3, std::placeholders::_4));
}

void Naviai_UDP_Communicate_App::onUDPRecv_(const std::uint8_t *data, size_t data_length, const std::string &sender_ip, int sender_port)
{
    if (!runFlag_)
        return;

    using namespace NAVIAI_UDP_Communication;
    auto unpacked_data = unpackData(data, data_length);
    for (auto &unpacked : unpacked_data)
    {
        auto res = std::async(std::launch::async, &Naviai_UDP_Communicate_App::onUDPDataRecv_, this, std::move(unpacked.data),
                              unpacked.packetType, unpacked.sendTimestamp, unpacked.recvTimestamp, sender_ip, sender_port);
        (void)res;
    }
}

bool getNumberIsNormal(const geometry_msgs::Vector3 &v)
{
    return (getNumberIsNormal(v.x) && getNumberIsNormal(v.y) && getNumberIsNormal(v.z));
}

bool getNumberIsNormal(const geometry_msgs::Twist &w)
{
    return (getNumberIsNormal(w.linear) && getNumberIsNormal(w.angular));
}

bool getNumberIsNormal(const robot_uplimb_pkg::WholeBodyPositionVelocity &target)
{
    return getNumberIsNormal(target.neckPosition) && getNumberIsNormal(target.neckVelocity) && getNumberIsNormal(target.leftArmPosition) &&
           getNumberIsNormal(target.leftArmVelocity) && getNumberIsNormal(target.rightArmPosition) &&
           getNumberIsNormal(target.rightArmVelocity) && getNumberIsNormal(target.leftHandPosition) &&
           getNumberIsNormal(target.leftHandVelocity) && getNumberIsNormal(target.rightHandPosition) &&
           getNumberIsNormal(target.rightHandVelocity) && getNumberIsNormal(target.waistPosition) &&
           getNumberIsNormal(target.waistVelocity);
}
