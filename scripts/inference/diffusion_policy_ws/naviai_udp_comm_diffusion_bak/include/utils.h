#ifndef UTILS_H
#define UTILS_H

#include <array>
#include <cmath>
#include <fstream>
#include <iostream>
#include <iterator>
#include <type_traits>
#include <yaml-cpp/yaml.h>

#if defined USING_NAVIAI_ROBOT_WA1_0303
static constexpr size_t BODY_JOINT_NUM = 19;
static const std::array<std::string, BODY_JOINT_NUM> jointNames = {
    "Shoulder_Y_R", "Shoulder_X_R", "Shoulder_Z_R", "Elbow_R", "Wrist_Z_R", "Wrist_Y_R", "Wrist_X_R",
    "Shoulder_Y_L", "Shoulder_X_L", "Shoulder_Z_L", "Elbow_L", "Wrist_Z_L", "Wrist_Y_L", "Wrist_X_L",
    "Neck_Z",       "Neck_Y",       "Waist_Z",      "Waist_Y", "Lifting_Z"};
static constexpr size_t LEFT_ARM_INDEX[2] = {7, 13};
static constexpr size_t RIGHT_ARM_INDEX[2] = {0, 6};
static constexpr size_t NECK_INDEX[2] = {14, 15};
static constexpr size_t WAIST_INDEX[2] = {16, 18};
#elif defined USING_NAVIAI_ROBOT_H1_PRO
static constexpr size_t BODY_JOINT_NUM = 17;
static const std::array<std::string, BODY_JOINT_NUM> jointNames = {
    "Shoulder_Y_R", "Shoulder_X_R", "Shoulder_Z_R", "Elbow_R",   "Wrist_Z_R", "Wrist_Y_R", "Wrist_X_R", "Shoulder_Y_L", "Shoulder_X_L",
    "Shoulder_Z_L", "Elbow_L",      "Wrist_Z_L",    "Wrist_Y_L", "Wrist_X_L", "Neck_Z",    "Neck_Y",    "A_Waist"};
static constexpr size_t LEFT_ARM_INDEX[2] = {7, 13};
static constexpr size_t RIGHT_ARM_INDEX[2] = {0, 6};
static constexpr size_t NECK_INDEX[2] = {14, 15};
static constexpr size_t WAIST_INDEX[2] = {16, 16};
#elif defined USING_NAVIAI_ROBOT_WA2_A2_LITE
static constexpr size_t BODY_JOINT_NUM = 22;
static const std::array<std::string, BODY_JOINT_NUM> jointNames = {
    "Shoulder_Z_L", "Shoulder_Y_L", "Shoulder_X_L", "Elbow_Z_L", "Elbow_Y_L", "Wrist_Z_L", "Wrist_Y_L", "Wrist_X_L",
    "Shoulder_Z_R", "Shoulder_Y_R", "Shoulder_X_R", "Elbow_Z_R", "Elbow_Y_R", "Wrist_Z_R", "Wrist_Y_R", "Wrist_X_R",
    "Neck_Z",       "Neck_Y",       "Pitch_Y_B",    "Pitch_Y_M", "Waist_Z",   "Waist_Y"};
static constexpr size_t LEFT_ARM_INDEX[2] = {0, 7};
static constexpr size_t RIGHT_ARM_INDEX[2] = {8, 15};
static constexpr size_t NECK_INDEX[2] = {16, 17};
static constexpr size_t WAIST_INDEX[2] = {18, 21};
#else
#error "Please define robot select macro: USING_NAVIAI_ROBOT_WA1_0303/USING_NAVIAI_ROBOT_H1_PRO/USING_NAVIAI_ROBOT_WA2_A2_LITE"
#endif
static constexpr size_t LEFT_ARM_JOINT_NUM = LEFT_ARM_INDEX[1] - LEFT_ARM_INDEX[0] + 1;
static constexpr size_t RIGHT_ARM_JOINT_NUM = RIGHT_ARM_INDEX[1] - RIGHT_ARM_INDEX[0] + 1;
static constexpr size_t NECK_JOINT_NUM = NECK_INDEX[1] - NECK_INDEX[0] + 1;
static constexpr size_t WAIST_JOINT_NUM = WAIST_INDEX[1] - WAIST_INDEX[0] + 1;
static constexpr size_t LEFT_HAND_JOINT_NUM = 6;
static constexpr size_t RIGHT_HAND_JOINT_NUM = 6;
static constexpr size_t JOINT_NUM = BODY_JOINT_NUM + LEFT_HAND_JOINT_NUM + RIGHT_HAND_JOINT_NUM;
static const std::array<std::string, LEFT_HAND_JOINT_NUM> handJointNames = {"thumb_MP",   "thumb_CMC", "index_MCP",
                                                                            "middle_MCP", "ring_MCP",  "little_MCP"};

template <typename T> typename std::enable_if<std::is_arithmetic<T>::value, double>::type inline RAD2DEG(T rad)
{
    return static_cast<double>(rad) * 180.0 / M_PI;
}

template <typename T> typename std::enable_if<std::is_arithmetic<T>::value, double>::type inline DEG2RAD(T deg)
{
    return static_cast<double>(deg) * M_PI / 180.0;
}

template <typename T> typename std::enable_if<std::is_arithmetic<T>::value, float>::type inline RAD2DEG_f(T rad)
{
    return static_cast<float>(rad * 180.0 / M_PI);
}

template <typename T> typename std::enable_if<std::is_arithmetic<T>::value, float>::type inline DEG2RAD_f(T deg)
{
    return static_cast<float>(deg * M_PI / 180.0);
}

#define RESET_TEXT "\033[0m"
#define RED_TEXT "\033[31m"
#define GREEN_TEXT "\033[32m"
#define YELLOW_TEXT "\033[33m"
#define BLUE_TEXT "\033[34m"
#define PURPLE_TEXT "\033[35m"
#define CYAN_TEXT "\033[36m"
#define WHITE_TEXT "\033[37m"

class LogStream
{
  public:
    enum class LogLevel
    {
        ERROR,
        WARN,
        INFO
    };

    LogStream(const char *file, int line, LogLevel level);
    ~LogStream();

    template <typename T> LogStream &operator<<(const T &val)
    {
        ss_ << val;
        return *this;
    }

  private:
    std::ostringstream ss_;
    const char *file_;
    int line_;
    LogLevel level_;
};

#define LOG_ERROR() LogStream(__FILE__, __LINE__, LogStream::LogLevel::ERROR)
#define LOG_WARN() LogStream(__FILE__, __LINE__, LogStream::LogLevel::WARN)
#define LOG_INFO() LogStream(__FILE__, __LINE__, LogStream::LogLevel::INFO)

typedef enum : size_t
{
    Neck_Target = 0,
    Left_Arm_Target = 1,
    Right_Arm_Target,
    Left_Hand_Target,
    Right_Hand_Target,
    Waist_Target,
    Target_All,
} Robot_Control_Type;

template <typename Container> std::string cvtDataString(const Container &container)
{
    std::ostringstream oss;
    oss << "[";
    size_t i = 0;
    for (const auto &item : container)
    {
        oss << item;
        if (++i < container.size())
            oss << ", ";
    }
    oss << "]";
    return oss.str();
}

template <typename T, std::size_t N> std::string cvtDataString(const std::array<T, N> &arr)
{
    std::ostringstream oss;
    oss << "[";
    for (size_t i = 0; i < N; ++i)
        oss << arr[i] << (i + 1 < N ? ", " : "");
    oss << "]";
    return oss.str();
}

template <typename T> std::string cvtDataString(const T *const vec, size_t len)
{
    std::ostringstream oss;
    oss << "Get invalid target in receive data: [";
    for (size_t i = 0; i < len; ++i)
        oss << vec[i] << (i + 1 < len ? ", " : "");
    oss << "]";
    return oss.str();
}

template <typename T> typename std::enable_if<std::is_arithmetic<T>::value, bool>::type getNumberIsNormal(T num)
{
    return !(std::isnan(num) || std::isinf(num));
}

template <typename Container>
typename std::enable_if<!std::is_arithmetic<Container>::value && std::is_same<decltype(std::begin(std::declval<Container>())),
                                                                              decltype(std::end(std::declval<Container>()))>::value,
                        bool>::type
getNumberIsNormal(const Container &container)
{
    for (const auto &item : container)
    {
        if (!getNumberIsNormal(item))
            return false;
    }
    return true;
}

class HandleControlFlags
{
  public:
    HandleControlFlags();
    virtual ~HandleControlFlags();
    void setControlFlag(Robot_Control_Type controlType, bool flag);
    bool getControlFlag(Robot_Control_Type controlType) const;
    void setControlValidFlag(std::uint8_t &validFlag, Robot_Control_Type type, bool set) const;
    bool getControlValidFlag(std::uint8_t flag, Robot_Control_Type type) const;

  private:
    std::array<bool, 4> controlFlags_;
};

extern YAML::Node loadRobotConfig(std::string &configPath);
extern void printConfig(YAML::Node config);
extern std::string cvtDataString(const std::vector<std::string> &vec);
extern double getUnixTimestampInSeconds(void);

#endif /* UTILS_H */
