#ifndef NAVIAI_UDP_PROTOCAL_H
#define NAVIAI_UDP_PROTOCAL_H

#include <cstdint>
#include <string>
#include <vector>

namespace NAVIAI_UDP_Communication
{
typedef enum : std::uint8_t
{
    Test_Latency = 0,
    Joint_States_Feedback = 1,
    Force_Feedback = 2,
    Packed_Target,
} Packet_Type_Enum;

typedef struct
{
    Packet_Type_Enum packetType;
    std::uint64_t sendTimestamp;
    std::uint64_t recvTimestamp;
    std::vector<std::uint8_t> data;
} Unpack_Data_Struct;

extern void debugPrintMultipackBufferState(void);

extern size_t getCurrentBufferMaxSize(void);
extern size_t getCurrentBufferUsedSize(void);
extern size_t getMultiFrameBufferUsedSize(void);
extern size_t getCurrentBufferOverflowCNT(void);
extern std::uint64_t getCRC32FailureCNT(void);

extern std::vector<std::vector<std::uint8_t>> packData(const std::uint8_t *data, size_t length, Packet_Type_Enum type);
extern std::vector<Unpack_Data_Struct> unpackData(const std::vector<std::uint8_t> &data);
extern std::vector<Unpack_Data_Struct> unpackData(const std::uint8_t *data, size_t length);
extern void clearBuffer(void);

extern std::string vectorToHexString(const std::vector<std::uint8_t> &vec);
}; // namespace NAVIAI_UDP_Communication

#endif /* NAVIAI_UDP_PROTOCAL_H */
