#include "communication/naviai_udp_protocal.h"

#include "crc32c/crc32c.h"
#include <algorithm>
#include <array>
#include <atomic>
#include <bits/stdint-uintn.h>
#include <chrono>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <stdexcept>
#include <tuple>
#include <type_traits>
#include <unordered_map>

namespace NAVIAI_UDP_Communication
{
#define PACKET_MAX_LENGTH 1024
#define PACKET_MAX_DATA_LENGTH (PACKET_MAX_LENGTH - sizeof(NAVIAI_UDP_Frame_Header) - sizeof(NAVIAI_UDP_Frame_Tail))
#define FRAME_HEADER 0x4E, 0x41, 0x56, 0x49, 0x41, 0x49, 0xFF
#define FRAME_TAIL 0x54, 0x45, 0x4C, 0x45, 0x4F, 0x50, 0x54, 0x4E

#define UNPACK_BUFFER_SIZE 10240
#define UNPACK_SUBPACKS_BUFFER_SIZE 5

struct __attribute__((packed)) NAVIAI_UDP_Frame_Header
{
    const std::uint8_t frameHeader[7] = {FRAME_HEADER}; // NAVIAI ASCII
    std::uint8_t packetType = 0;
    // 8 Bytes
    std::uint16_t packetLength = 0;
    std::uint16_t packetID = 0;
    std::uint16_t subpackNum = 0;
    std::uint16_t subpackID = 0;
    // 16 Bytes
    std::uint32_t timestamp = 0;
    std::uint32_t crcNumber = 0;
}; // 24 Bytes

struct __attribute__((packed)) NAVIAI_UDP_Frame_Tail
{
    const std::uint8_t frameTail[8] = {FRAME_TAIL}; // TELEOPTN ASCII
};                                                   // 8 Bytes

union Header_Union
{
    NAVIAI_UDP_Frame_Header headerStruct;
    std::uint8_t headerData[sizeof(NAVIAI_UDP_Frame_Header)];
};

union Tail_Union
{
    NAVIAI_UDP_Frame_Tail tailStruct;
    std::uint8_t tailData[sizeof(NAVIAI_UDP_Frame_Tail)];
};

struct Unpack_Frame_Result
{
    bool valid;
    size_t unpackLength;
    Packet_Type_Enum type;
    std::uint16_t packetID;
    std::uint16_t subpackNum;
    std::uint16_t subpackID;
    std::uint64_t recvTimestamp;
    std::uint64_t sendTimestamp;
    std::vector<std::uint8_t> data;
};

class CircularQueue
{
  public:
    CircularQueue() : front_(0), rear_(0), size_(0), overflowCount_(0)
    {
        buffer_.fill(0);
    }

    virtual ~CircularQueue()
    {
    }

    void enqueue(const std::uint8_t *data, size_t length)
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (length > UNPACK_BUFFER_SIZE)
        {
            size_t writeLength = UNPACK_BUFFER_SIZE;
            std::cout << "Data length exceeds buffer size. Wrapping around to write " << writeLength << " bytes." << std::endl;
            length = writeLength;
            data += (length - writeLength);
            size_ = 0;
            front_ = 0;
            ++overflowCount_;
        }
        if (length > remainingSpaceNoLock_())
        {
            printf("Not enough remaining space %ld bytes since is writing size of %ld bytes.\n", remainingSpaceNoLock_(), length);
            ++overflowCount_;
            front_ = (front_ + length - remainingSpaceNoLock_()) % UNPACK_BUFFER_SIZE;
        }
        size_t spaceToEnd = UNPACK_BUFFER_SIZE - rear_;
        if (length <= spaceToEnd)
        {
            std::memcpy(&buffer_[rear_], data, length);
            rear_ = (rear_ + length) % UNPACK_BUFFER_SIZE;
        }
        else
        {
            std::memcpy(&buffer_[rear_], data, spaceToEnd);
            std::memcpy(&buffer_[0], data + spaceToEnd, length - spaceToEnd);
            rear_ = length - spaceToEnd;
        }
        size_ += length;
        if (size_ > UNPACK_BUFFER_SIZE)
        {
            printf("Buffer overflow! Adjusting size to %d bytes.\n", UNPACK_BUFFER_SIZE);
            ++overflowCount_;
            size_ = UNPACK_BUFFER_SIZE;
        }
    }

    void dequeue(size_t length)
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (length > size_)
        {
            printf("Not enough data to dequeue.\n");
            front_ = 0;
            rear_ = 0;
            size_ = 0;
        }
        front_ = (front_ + length) % UNPACK_BUFFER_SIZE;
        size_ -= length;
    }

    ssize_t findSubstring(const std::uint8_t *sub_data, size_t sub_len) const
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (sub_len > size_)
            return -1;
        for (size_t i = 0; i <= size_ - sub_len; ++i)
        {
            size_t idx = (front_ + i) % UNPACK_BUFFER_SIZE;
            if (idx + sub_len <= UNPACK_BUFFER_SIZE)
            {
                if (std::memcmp(&buffer_[idx], sub_data, sub_len) == 0)
                    return i;
            }
            else
            {
                // Handle the wraparound case: compare first part from idx to the end of buffer
                size_t space_to_end = UNPACK_BUFFER_SIZE - idx;
                if (std::memcmp(&buffer_[idx], sub_data, space_to_end) == 0 &&
                    std::memcmp(&buffer_[0], sub_data + space_to_end, sub_len - space_to_end) == 0)
                    return i;
            }
        }
        return -1;
    }

    ssize_t findSubstringKmp(const std::uint8_t *subData, size_t subLen) const
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (subLen > size_)
            return -1;
        // Generate the "next" array for KMP
        std::vector<int> next = calculateKmpNextArray(subData, subLen);
        // Perform the KMP search
        int matched = 0;
        for (size_t i = 0; i < size_; ++i)
        {
            size_t idx = (front_ + i) % UNPACK_BUFFER_SIZE;
            while (matched > 0 && subData[matched] != buffer_[idx])
                matched = next[matched - 1];
            if (subData[matched] == buffer_[idx])
            {
                ++matched;
                if (matched == subLen)
                    return i - subLen + 1;
            }
            else
                matched = 0;
        }
        return -1;
    }

    ssize_t findSubstringKmpGivenNextArray(const std::uint8_t *subData, size_t subLen, std::vector<int> &next, size_t startIndex = 0) const
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if ((subLen > size_) || (startIndex > size_) || (subLen > size_ - startIndex))
            return -1;
        // Perform the KMP search
        int matched = 0;
        for (size_t i = startIndex; i < size_; ++i)
        {
            size_t idx = (front_ + i) % UNPACK_BUFFER_SIZE;
            while (matched > 0 && subData[matched] != buffer_[idx])
                matched = next[matched - 1];
            if (subData[matched] == buffer_[idx])
            {
                ++matched;
                if (matched == subLen)
                    return i - subLen + 1;
            }
            else
                matched = 0;
        }
        return -1;
    }

    std::vector<int> calculateKmpNextArray(const std::uint8_t *subData, size_t subLen) const
    {
        std::vector<int> next(subLen, 0);
        int j = 0;
        for (int i = 1; i < subLen; ++i)
        {
            while (j > 0 && subData[i] != subData[j])
                j = next[j - 1];
            if (subData[i] == subData[j])
                ++j;
            next[i] = j;
        }
        return next;
    }

    std::uint8_t &at(size_t index)
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (index >= size_)
            throw std::runtime_error("Index " + std::to_string(index) + " out of range " + std::to_string(size_) + ".");
        return buffer_[(front_ + index) % UNPACK_BUFFER_SIZE];
    }

    size_t currentSize(void) const
    {
        std::lock_guard<std::mutex> lock(mutex_);
        return size_;
    }

    size_t remainingSpace(void) const
    {
        std::lock_guard<std::mutex> lock(mutex_);
        return UNPACK_BUFFER_SIZE - size_;
    }

    std::vector<std::uint8_t> toVector(void) const
    {
        return toVector(0, size_ - 1);
    }

    std::vector<std::uint8_t> toVector(size_t startIndex, size_t endIndex) const
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (startIndex > endIndex || endIndex >= size_)
            throw std::runtime_error("Invalid index range.");
        std::vector<std::uint8_t> result;
        result.reserve(endIndex - startIndex + 1);
        for (size_t i = startIndex; i <= endIndex; ++i)
        {
            size_t idx = (front_ + i) % UNPACK_BUFFER_SIZE;
            result.push_back(buffer_[idx]);
        }
        return result;
    }

    void printQueue(void) const
    {
        std::lock_guard<std::mutex> lock(mutex_);
        for (size_t i = 0; i < size_; ++i)
        {
            size_t idx = (front_ + i) % UNPACK_BUFFER_SIZE;
            std::cout << std::hex << "0x" << (int)buffer_[idx] << " ";
        }
        std::cout << std::dec << std::endl;
    }

    size_t getOverlflowCount(void) const
    {
        std::lock_guard<std::mutex> lock(mutex_);
        return overflowCount_;
    }

  private:
    std::array<std::uint8_t, UNPACK_BUFFER_SIZE> buffer_;
    size_t front_;
    size_t rear_;
    size_t size_;
    size_t overflowCount_;
    mutable std::mutex mutex_;

    size_t remainingSpaceNoLock_(void) const
    {
        return UNPACK_BUFFER_SIZE - size_;
    }
};

struct Multi_Frame_Unpack_Buffer
{
    bool receiving;
    std::uint32_t packetID;
    size_t validPackNum;
    size_t totalPackNum;
    Unpack_Data_Struct unpackData;
    std::vector<std::vector<std::uint8_t>> data;
};

static const std::array<std::uint8_t, 7> frameHeader_ = {FRAME_HEADER};
static const std::array<std::uint8_t, 8> frameTail_ = {FRAME_TAIL};

static std::mutex packMutex_;
static std::uint16_t packetID_ = 0;

static std::atomic<std::uint64_t> unpackCRCFailureCNT_{0};

static CircularQueue unpackBuffer_;
static std::vector<int> kmpNextFrameHeader_;
static std::vector<int> kmpNextFrameTail_;
static bool multiFrameBufferInited_ = false;
static std::vector<Multi_Frame_Unpack_Buffer> multiFrameBuffer_;
static std::unordered_map<std::uint32_t, size_t> frameID2IndexMap_;
static std::vector<std::unique_ptr<std::mutex>> multiFrameBufferMutex_;
static std::mutex unpackAllLock_;

size_t getCurrentBufferMaxSize(void)
{
    return UNPACK_BUFFER_SIZE;
}

size_t getCurrentBufferUsedSize(void)
{
    return unpackBuffer_.currentSize();
}

size_t getCurrentBufferOverflowCNT(void)
{
    return unpackBuffer_.getOverlflowCount();
}

size_t getMultiFrameBufferUsedSize(void)
{
    size_t used_size = 0;
    for (size_t i = 0; i < multiFrameBuffer_.size(); ++i)
    {
        const Multi_Frame_Unpack_Buffer &buffer = multiFrameBuffer_.at(i);
        std::lock_guard<std::mutex> lock(*multiFrameBufferMutex_.at(i));
        if (buffer.receiving)
            for (const auto &data : buffer.data)
                used_size += data.size();
    }
    return used_size;
}

std::uint64_t getCRC32FailureCNT(void)
{
    return unpackCRCFailureCNT_;
}

static std::uint64_t getCurrentUnixTimestampUs(void)
{
    using namespace std::chrono;
    auto now = system_clock::now();
    auto duration = now.time_since_epoch();
    return duration_cast<microseconds>(duration).count();
}

void debugPrintMultipackBufferState(void)
{
    printf("MultiBuffer State: Overall buffer size: %ld bytes. Each buffer state:\n", getMultiFrameBufferUsedSize());
    for (size_t i = 0; i < UNPACK_SUBPACKS_BUFFER_SIZE; ++i)
        printf("\tbuffer %ld: receiving: %d, frame id: %u, received packs: %ld / %ld\n", i, multiFrameBuffer_[i].receiving,
               multiFrameBuffer_[i].packetID, multiFrameBuffer_[i].validPackNum, multiFrameBuffer_[i].totalPackNum);
}

std::vector<std::vector<std::uint8_t>> packData(const std::uint8_t *data, size_t length, Packet_Type_Enum type)
{
    static_assert(PACKET_MAX_DATA_LENGTH >= 0, "Error: PACKET_MAX_DATA_LENGTH is negative!");
    size_t frameNum = (length + PACKET_MAX_DATA_LENGTH - 1) / PACKET_MAX_DATA_LENGTH;
    std::uint16_t getPacketID;
    {
        std::lock_guard<std::mutex> lock(packMutex_);
        getPacketID = packetID_;
        packetID_ += 1;
    }
    auto pack = [type, getPacketID](const std::uint8_t *packet_data, size_t len, std::uint16_t subpackNum, std::uint16_t subpackID,
                                    std::vector<std::uint8_t> &packed) -> void {
        NAVIAI_UDP_Frame_Header header;
        NAVIAI_UDP_Frame_Tail tail;
        size_t packetLength = sizeof(header) + len + sizeof(tail);
        packed.resize(packetLength);
        header.packetType = static_cast<std::uint8_t>(type);
        header.packetLength = packetLength;
        header.packetID = getPacketID;
        header.subpackNum = subpackNum;
        header.subpackID = subpackID;
        header.timestamp = static_cast<std::uint32_t>(getCurrentUnixTimestampUs());
        header.crcNumber = crc32c::Crc32c(packet_data, len);
        std::memcpy(packed.data(), &header, sizeof(header));
        std::memcpy(packed.data() + sizeof(header), packet_data, len);
        std::memcpy(packed.data() + sizeof(header) + len, &tail, sizeof(tail));
    };
    std::vector<std::vector<std::uint8_t>> result_frames;
    result_frames.reserve(frameNum);
    for (size_t i = 0; i < frameNum; ++i)
    {
        result_frames.push_back(std::vector<std::uint8_t>());
        if (i < frameNum - 1)
            pack(data + i * PACKET_MAX_DATA_LENGTH, PACKET_MAX_DATA_LENGTH, frameNum, i, result_frames.at(i));
        else
            pack(data + i * PACKET_MAX_DATA_LENGTH, length % PACKET_MAX_DATA_LENGTH, frameNum, i, result_frames.at(i));
    }
    return result_frames;
}

static std::uint64_t inferTimestamp_(std::uint64_t now, std::uint32_t toInfer)
{
    std::uint64_t highBits = now & 0xFFFFFFFF00000000ULL;
    std::uint64_t candidate1 = highBits | toInfer;
    std::uint64_t candidate2 = (highBits + 0x100000000ULL) | toInfer;
    std::uint64_t candidate3 = (highBits - 0x100000000ULL) | toInfer;
    std::uint64_t diff1 = (candidate1 > now) ? (candidate1 - now) : (now - candidate1);
    std::uint64_t diff2 = (candidate2 > now) ? (candidate2 - now) : (now - candidate2);
    std::uint64_t diff3 = (candidate3 > now) ? (candidate3 - now) : (now - candidate3);
    if (diff1 <= diff2 && diff1 <= diff3)
        return candidate1;
    else if (diff2 <= diff1 && diff2 <= diff3)
        return candidate2;
    else
        return candidate3;
}

static void initMultiframeBuffer_(void);
static int getMultipackBufferFirstUnusedIndex_(void)
{
    if (!multiFrameBufferInited_)
        initMultiframeBuffer_();
    for (size_t i = 0; i < multiFrameBuffer_.size(); ++i)
        if (!multiFrameBuffer_.at(i).receiving)
            return i;
    return -1;
}

static int getMultipackBufferEarliestReceivedFrame_(void)
{
    if (!multiFrameBufferInited_)
        initMultiframeBuffer_();
    std::uint64_t minRecvTime = static_cast<std::uint64_t>(-1);
    int frameIndex = -1;
    for (size_t i = 0; i < multiFrameBuffer_.size(); ++i)
    {
        if (multiFrameBuffer_.at(i).receiving && (multiFrameBuffer_.at(i).unpackData.recvTimestamp < minRecvTime))
        {
            minRecvTime = multiFrameBuffer_.at(i).unpackData.recvTimestamp;
            frameIndex = i;
        }
    }
    return frameIndex;
}

static void reinitMultipackBuffer_(Multi_Frame_Unpack_Buffer &buffer)
{
    frameID2IndexMap_.erase(buffer.packetID);
    buffer.packetID = buffer.unpackData.recvTimestamp = buffer.unpackData.sendTimestamp = buffer.totalPackNum = buffer.validPackNum = 0;
    for (auto &buffer : buffer.data)
        buffer.clear();
    buffer.data.clear();
    buffer.unpackData.packetType = static_cast<Packet_Type_Enum>(0);
    buffer.unpackData.data.clear();
    buffer.receiving = false;
}

static void initMultiframeBuffer_(void)
{
    if (!multiFrameBufferInited_)
    {
        multiFrameBuffer_.reserve(UNPACK_SUBPACKS_BUFFER_SIZE);
        multiFrameBufferMutex_.reserve(UNPACK_SUBPACKS_BUFFER_SIZE);
        for (size_t i = 0; i < UNPACK_SUBPACKS_BUFFER_SIZE; ++i)
        {
            multiFrameBuffer_.emplace_back(Multi_Frame_Unpack_Buffer());
            multiFrameBufferMutex_.emplace_back(std::make_unique<std::mutex>());
        }
    }
    multiFrameBufferInited_ = true;
    for (auto &buffer : multiFrameBuffer_)
        reinitMultipackBuffer_(buffer);
    frameID2IndexMap_.clear();
}

static std::vector<Unpack_Data_Struct> handleMultipacks_(Unpack_Frame_Result &&result)
{
    std::vector<Unpack_Data_Struct> res;
    auto it = frameID2IndexMap_.find(result.packetID);
    if (!multiFrameBufferInited_)
        initMultiframeBuffer_();
    if (it != frameID2IndexMap_.end())
    {
        Multi_Frame_Unpack_Buffer &buffer = multiFrameBuffer_.at(it->second);
        std::lock_guard<std::mutex> lock(*multiFrameBufferMutex_.at(it->second));
        buffer.receiving = true;
        if ((buffer.totalPackNum != result.subpackNum) || (buffer.unpackData.packetType != result.type))
        {
            reinitMultipackBuffer_(buffer);
            return handleMultipacks_(std::move(result));
        }
        if (result.subpackID >= buffer.totalPackNum)
        {
            printf("Get invalid subpack index %d of total %ld.", result.subpackID, buffer.totalPackNum);
            return res;
        }
        buffer.unpackData.recvTimestamp = std::max(result.recvTimestamp, buffer.unpackData.recvTimestamp);
        buffer.unpackData.sendTimestamp = std::max(result.sendTimestamp, buffer.unpackData.sendTimestamp);
        if (buffer.data.at(result.subpackID).size() == 0)
        {
            buffer.validPackNum += 1;
            buffer.data.at(result.subpackID) = std::move(result.data);
        }
        else
            buffer.data.at(result.subpackID) = std::move(result.data);
        if (buffer.validPackNum == buffer.totalPackNum)
        {
            for (auto &vec : buffer.data)
                buffer.unpackData.data.insert(buffer.unpackData.data.end(), std::make_move_iterator(vec.begin()),
                                              std::make_move_iterator(vec.end()));
            res.push_back(std::move(buffer.unpackData));
            reinitMultipackBuffer_(buffer);
        }
    }
    else
    {
        auto writeNewBuffer = [&result](Multi_Frame_Unpack_Buffer &buffer, size_t index) -> void {
            reinitMultipackBuffer_(multiFrameBuffer_.at(index));
            std::lock_guard<std::mutex> lock(*multiFrameBufferMutex_.at(index));
            buffer.packetID = result.packetID;
            buffer.receiving = true;
            buffer.totalPackNum = result.subpackNum;
            buffer.validPackNum = 1;
            buffer.unpackData.packetType = result.type;
            buffer.unpackData.sendTimestamp = result.sendTimestamp;
            buffer.unpackData.recvTimestamp = result.recvTimestamp;
            buffer.data.resize(result.subpackNum);
            buffer.data.at(result.subpackID) = std::move(result.data);
            frameID2IndexMap_[result.packetID] = index;
        };
        int index = getMultipackBufferFirstUnusedIndex_();
        if (index >= 0)
        {
            if (result.subpackID >= result.subpackNum)
            {
                printf("Get invalid subpack index %d of total %d.\n", result.subpackID, result.subpackNum);
                return res;
            }
            writeNewBuffer(multiFrameBuffer_.at(index), index);
        }
        else
        {
            index = getMultipackBufferEarliestReceivedFrame_();
            if ((index < 0) || (index > UNPACK_SUBPACKS_BUFFER_SIZE))
            {
                printf("Get index %d, Couldn't get valid buffer to write!\n", index);
                return res;
            }
            writeNewBuffer(multiFrameBuffer_.at(index), index);
        }
    }
    return res;
}

static Unpack_Frame_Result unpackWholeFrame_(const std::uint8_t *data, size_t length)
{
    Unpack_Frame_Result result;
    if ((length < sizeof(NAVIAI_UDP_Frame_Header) + sizeof(NAVIAI_UDP_Frame_Tail)) ||
        (!std::equal(frameHeader_.begin(), frameHeader_.end(), data)))
    {
        result.valid = false;
        result.unpackLength = 0;
        return result;
    }
    const Header_Union *header = reinterpret_cast<const Header_Union *>(data);
    auto packetLength = header->headerStruct.packetLength;
    if (!std::equal(frameTail_.begin(), frameTail_.end(), data + packetLength - sizeof(NAVIAI_UDP_Frame_Tail)))
    {
        result.valid = false;
        result.unpackLength = 0;
        return result;
    }
    result.unpackLength = packetLength;
    std::uint32_t calculated_crc32 = crc32c::Crc32c(data + sizeof(NAVIAI_UDP_Frame_Header),
                                                    packetLength - sizeof(NAVIAI_UDP_Frame_Tail) - sizeof(NAVIAI_UDP_Frame_Header));
    if (calculated_crc32 != header->headerStruct.crcNumber)
    {
        unpackCRCFailureCNT_ += 1;
        result.valid = false;
        return result;
    }
    result.valid = true;
    result.type = static_cast<Packet_Type_Enum>(header->headerStruct.packetType);
    result.packetID = header->headerStruct.packetID;
    result.subpackNum = header->headerStruct.subpackNum;
    result.subpackID = header->headerStruct.subpackID;
    result.recvTimestamp = getCurrentUnixTimestampUs();
    result.sendTimestamp = inferTimestamp_(result.recvTimestamp, header->headerStruct.timestamp);
    result.data = std::vector<std::uint8_t>(data + sizeof(NAVIAI_UDP_Frame_Header), data + packetLength - sizeof(NAVIAI_UDP_Frame_Tail));
    return result;
}

static Unpack_Frame_Result unpackWholeFrame_(const std::vector<std::uint8_t> &data)
{
    return unpackWholeFrame_(data.data(), data.size());
}

static std::vector<Unpack_Data_Struct> unpackFromBuffer_(bool &continueFlag)
{
    std::vector<Unpack_Data_Struct> unpackedDataVec;
    if ((kmpNextFrameHeader_.size() == 0) || (kmpNextFrameTail_.size() == 0))
    {
        kmpNextFrameHeader_ = unpackBuffer_.calculateKmpNextArray(frameHeader_.data(), frameHeader_.size());
        kmpNextFrameTail_ = unpackBuffer_.calculateKmpNextArray(frameTail_.data(), frameTail_.size());
    }
    // Find frame header
    auto index = unpackBuffer_.findSubstringKmpGivenNextArray(frameHeader_.data(), frameHeader_.size(), kmpNextFrameHeader_);
    if (index == -1)
    {
        continueFlag = false;
        return unpackedDataVec;
    }
    // Unpack
    Unpack_Frame_Result unpackedData = unpackWholeFrame_(unpackBuffer_.toVector(index, unpackBuffer_.currentSize() - 1));
    if (unpackedData.valid)
        unpackBuffer_.dequeue(unpackedData.unpackLength);
    while (!unpackedData.valid)
    {
        auto index_test = index;
        index = unpackBuffer_.findSubstringKmpGivenNextArray(frameHeader_.data(), frameHeader_.size(), kmpNextFrameHeader_, index_test + 1);
        if (index == -1)
        {
            continueFlag = false;
            return unpackedDataVec;
        }
        unpackedData = unpackWholeFrame_(unpackBuffer_.toVector(index, unpackBuffer_.currentSize() - 1));
        if (unpackedData.valid)
            unpackBuffer_.dequeue(index + unpackedData.unpackLength);
    }
    if (unpackedData.valid)
    {
        if (unpackedData.subpackNum == 1)
        {
            Unpack_Data_Struct newUnpackedData;
            newUnpackedData.packetType = unpackedData.type;
            newUnpackedData.sendTimestamp = unpackedData.sendTimestamp;
            newUnpackedData.recvTimestamp = unpackedData.recvTimestamp;
            newUnpackedData.data = std::move(unpackedData.data);
            unpackedDataVec.push_back(std::move(newUnpackedData));
        }
        else
        {
            std::vector<Unpack_Data_Struct> res = handleMultipacks_(std::move(unpackedData));
            if (!res.empty())
            {
                unpackedDataVec.insert(unpackedDataVec.end(), std::make_move_iterator(res.begin()), std::make_move_iterator(res.end()));
                res.clear();
            }
        }
        continueFlag = true;
        return unpackedDataVec;
    }
    continueFlag = false;
    return unpackedDataVec;
}

std::vector<Unpack_Data_Struct> unpackDataSingleWithMaxSize_(const std::uint8_t *data, size_t length)
{
    std::vector<Unpack_Data_Struct> res;
    // Push data to buffer
    unpackBuffer_.enqueue(data, length);
    // Unpack from buffer
    bool continueFlag = true;
    std::vector<Unpack_Data_Struct> unpackedData = unpackFromBuffer_(continueFlag);
    while (continueFlag)
    {
        if (!unpackedData.empty())
            res.insert(res.end(), std::make_move_iterator(unpackedData.begin()), std::make_move_iterator(unpackedData.end()));
        unpackedData = unpackFromBuffer_(continueFlag);
    }
    return res;
}

std::vector<Unpack_Data_Struct> unpackData(const std::uint8_t *data, size_t length)
{
    std::vector<Unpack_Data_Struct> res;
    constexpr size_t maxSingleDataSize = UNPACK_BUFFER_SIZE / 2;
    {
        std::lock_guard<std::mutex> lock(unpackAllLock_);
        if (length < maxSingleDataSize)
            return unpackDataSingleWithMaxSize_(data, length);
        size_t num = (length + maxSingleDataSize - 1) / maxSingleDataSize;
        for (size_t i = 0; i < num - 1; ++i)
        {
            std::vector<Unpack_Data_Struct> tmp = unpackDataSingleWithMaxSize_(data + i * maxSingleDataSize, maxSingleDataSize);
            if (!tmp.empty())
                res.insert(res.end(), std::make_move_iterator(tmp.begin()), std::make_move_iterator(tmp.end()));
        }
        std::vector<Unpack_Data_Struct> tmp =
            unpackDataSingleWithMaxSize_(data + (num - 1) * maxSingleDataSize, length % maxSingleDataSize);
        if (!tmp.empty())
            res.insert(res.end(), std::make_move_iterator(tmp.begin()), std::make_move_iterator(tmp.end()));
    }
    return res;
}

std::vector<Unpack_Data_Struct> unpackData(const std::vector<std::uint8_t> &data)
{
    return unpackData(data.data(), data.size());
}

void clearBuffer(void)
{
    unpackBuffer_.dequeue(unpackBuffer_.currentSize());
    initMultiframeBuffer_();
}

std::string vectorToHexString(const std::vector<std::uint8_t> &vec)
{
    std::stringstream ss;
    for (auto byte : vec)
        ss << "0x" << std::setw(2) << std::setfill('0') << std::hex << (int)byte << " ";
    return ss.str();
}
} // namespace NAVIAI_UDP_Communication
