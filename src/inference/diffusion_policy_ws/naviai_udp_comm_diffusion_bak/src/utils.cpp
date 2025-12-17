#include "utils.h"

#include <chrono>
#include <iomanip>

LogStream::LogStream(const char *file, int line, LogLevel level) : file_(file), line_(line), level_(level)
{
}

LogStream::~LogStream()
{
    switch (level_)
    {
    case LogLevel::ERROR:
        std::cerr << RED_TEXT << "[ERROR][";
        break;
    case LogLevel::INFO:
        std::cerr << WHITE_TEXT << "[INFO][";
        break;
    case LogLevel::WARN:
        std::cerr << YELLOW_TEXT << "[WARN][";
        break;
    default:
        std::cerr << CYAN_TEXT << "[UNKOWN][";
        break;
    }
    std::cerr << std::fixed << std::setprecision(6) << getUnixTimestampInSeconds() << "][" << file_ << ":" << line_ << "] " << ss_.str()
              << RESET_TEXT << std::endl;
}

YAML::Node loadRobotConfig(std::string &configPath)
{
    std::ifstream file(configPath);
    YAML::Node config;
    if (!file)
    {
        LOG_ERROR() << "Failed to open config file: " << configPath;
        return config;
    }
    try
    {
        config = YAML::Load(file);
    }
    catch (const YAML::Exception &e)
    {
        LOG_ERROR() << "Failed to parse YAML file: " << e.what();
    }
    return config;
}

void printConfig(YAML::Node config)
{
    if (config)
        LOG_INFO() << "Loaded config: " << config;
}

std::string cvtDataString(const std::vector<std::string> &vec)
{
    std::ostringstream oss;
    for (const auto &str : vec)
        oss << str << " ";
    return oss.str();
}

double getUnixTimestampInSeconds(void)
{
    using namespace std::chrono;
    auto duration = system_clock::now().time_since_epoch();
    return duration_cast<nanoseconds>(duration).count() / 1e9;
}

HandleControlFlags::HandleControlFlags()
{
    controlFlags_.fill(0);
}

HandleControlFlags::~HandleControlFlags()
{
    controlFlags_.fill(0);
}

void HandleControlFlags::setControlFlag(Robot_Control_Type controlType, bool flag)
{
    switch (controlType)
    {
    case Neck_Target:
        controlFlags_[0] = flag;
        return;
    case Left_Arm_Target:
    case Right_Arm_Target:
        controlFlags_[1] = flag;
        return;
    case Left_Hand_Target:
    case Right_Hand_Target:
        controlFlags_[2] = flag;
        return;
    case Waist_Target:
        controlFlags_[3] = flag;
        return;
    default:
        return;
    }
}

bool HandleControlFlags::getControlFlag(Robot_Control_Type controlType) const
{
    switch (controlType)
    {
    case Neck_Target:
        return controlFlags_[0];
    case Left_Arm_Target:
    case Right_Arm_Target:
        return controlFlags_[1];
    case Left_Hand_Target:
    case Right_Hand_Target:
        return controlFlags_[2];
    case Waist_Target:
        return controlFlags_[3];
    default:
        return false;
    }
}

void HandleControlFlags::setControlValidFlag(std::uint8_t &validFlag, Robot_Control_Type type, bool set) const
{
    std::uint8_t operateNum = 0;
    switch (type)
    {
    case Neck_Target:
        operateNum = (1 << 5);
        break;
    case Left_Arm_Target:
    case Right_Arm_Target:
        operateNum = (0b11 << 3);
        break;
    case Left_Hand_Target:
    case Right_Hand_Target:
        operateNum = (0b11 << 1);
        break;
    case Waist_Target:
        operateNum = (1 << 0);
        break;
    default:
        return;
    }
    if (set)
        validFlag |= operateNum;
    else
        validFlag &= ~operateNum;
}

bool HandleControlFlags::getControlValidFlag(std::uint8_t flag, Robot_Control_Type type) const
{
    switch (type)
    {
    case Neck_Target:
        return flag & (1 << 5);
    case Left_Arm_Target:
    case Right_Arm_Target:
        return flag & (0b11 << 3);
    case Left_Hand_Target:
    case Right_Hand_Target:
        return flag & (0b11 << 1);
    case Waist_Target:
        return flag & (1 << 0);
    default:
        return false;
    }
}
