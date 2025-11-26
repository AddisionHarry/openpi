#ifndef CONTROL_TARGET_BUFFER_HPP
#define CONTROL_TARGET_BUFFER_HPP

#include "../utils.h"

#include <array>
#include <cassert>
#include <cstddef>
#include <functional>
#include <memory>
#include <mutex>
#include <vector>

template <typename T> class IControlTargetBufferBase
{
  public:
    virtual void pushData(const T *newData, bool valid) = 0;
    virtual void pushData(const std::vector<T> &newData, bool valid) = 0;
    virtual const T *getDataPtr(void) const = 0;
    virtual void editCurrentData(std::function<void(T const *)> editFunc) = 0;
    virtual bool getValid(void) const = 0;
    virtual double getUpdateTime(void) const = 0;
    virtual bool getUpdated(void) const
    {
        return false;
    }
    virtual void switchBuffer(void)
    {
    }
};

template <typename T, size_t N> class ControlTargetSingleBuffer : public IControlTargetBufferBase<T>
{
  public:
    ControlTargetSingleBuffer() : valid_(false)
    {
        buffer_.fill(T(0));
    }

    void pushData(const std::vector<T> &newData, bool valid) override
    {
        if (newData.size() != N)
            return;
        pushData(newData.data(), valid);
        updateTime_ = getUnixTimestampInSeconds();
    }

    void pushData(const T *newData, bool valid) override
    {
        if (!newData)
            return;
        std::lock_guard<std::mutex> lock(mutex_);
        std::copy(newData, newData + N, buffer_.begin());
        valid_ = valid;
        updateTime_ = getUnixTimestampInSeconds();
    }

    void editCurrentData(std::function<void(T const *)> editFunc) override
    {
        std::lock_guard<std::mutex> lock(mutex_);
        editFunc(buffer_.data());
        updateTime_ = getUnixTimestampInSeconds();
    }

    const T *getDataPtr(void) const override
    {
        return buffer_.data();
    }

    bool getValid(void) const override
    {
        return valid_;
    }

    double getUpdateTime(void) const override
    {
        return updateTime_;
    }

  private:
    std::array<T, N> buffer_;
    bool valid_;
    double updateTime_;
    mutable std::mutex mutex_;
};

template <typename T, size_t N> class ControlTargetDoubleBuffer : public IControlTargetBufferBase<T>
{
  public:
    ControlTargetDoubleBuffer() : currentBuffer_(0), dataUpdated_{false, false}, valid_{false, false}
    {
        for (auto &buffer : buffers_)
            buffer.fill(T(0));
    }

    void pushData(const T *newData, bool valid) override
    {
        if (!newData)
            return;
        std::lock_guard<std::mutex> lock(mutex_);
        size_t next = 1 - currentBuffer_;
        std::copy(newData, newData + N, buffers_.at(next).begin());
        valid_.at(next) = valid;
        dataUpdated_.at(next) = true;
        updateTime_.at(next) = getUnixTimestampInSeconds();
    }

    void pushData(const std::vector<T> &newData, bool valid) override
    {
        if (newData.size() != N)
            return;
        pushData(newData.data(), valid);
    }

    void editCurrentData(std::function<void(T const *)> editFunc) override
    {
        std::lock_guard<std::mutex> lock(mutex_);
        editFunc(buffers_.at(currentBuffer_).data());
        updateTime_.at(currentBuffer_) = getUnixTimestampInSeconds();
    }

    const T *getDataPtr(void) const override
    {
        return buffers_.at(currentBuffer_).data();
    }

    bool getValid(void) const override
    {
        return valid_.at(currentBuffer_);
    }

    bool getUpdated(void) const override
    {
        return dataUpdated_.at(1 - currentBuffer_);
    }

    void switchBuffer(void) override
    {
        std::lock_guard<std::mutex> lock(mutex_);
        size_t next = 1 - currentBuffer_;
        if (!dataUpdated_.at(next))
            return;
        currentBuffer_ = next;
        dataUpdated_.at(currentBuffer_) = false;
        valid_.at(1 - currentBuffer_) = false;
    }

    double getUpdateTime(void) const override
    {
        return updateTime_.at(currentBuffer_);
    }

  private:
    std::array<std::array<T, N>, 2> buffers_;
    std::array<bool, 2> valid_{false, false};
    std::array<bool, 2> dataUpdated_{false, false};
    std::array<double, 2> updateTime_{0.0, 0.0};
    size_t currentBuffer_;
    mutable std::mutex mutex_;
};

template <typename T> class ControlTargetBuffer
{
  public:
    explicit ControlTargetBuffer(bool useDoubleBuffer)
    {
        if (useDoubleBuffer)
        {
            neckBuffer_ = std::make_unique<ControlTargetDoubleBuffer<T, NECK_JOINT_NUM * 2>>();
            waistBuffer_ = std::make_unique<ControlTargetDoubleBuffer<T, WAIST_JOINT_NUM * 2>>();
            armBuffer_[0] = std::make_unique<ControlTargetDoubleBuffer<T, LEFT_ARM_JOINT_NUM * 2>>();
            armBuffer_[1] = std::make_unique<ControlTargetDoubleBuffer<T, LEFT_ARM_JOINT_NUM * 2>>();
            handBuffer_[0] = std::make_unique<ControlTargetDoubleBuffer<T, LEFT_HAND_JOINT_NUM * 2>>();
            handBuffer_[1] = std::make_unique<ControlTargetDoubleBuffer<T, RIGHT_HAND_JOINT_NUM * 2>>();
        }
        else
        {
            neckBuffer_ = std::make_unique<ControlTargetSingleBuffer<T, NECK_JOINT_NUM * 2>>();
            waistBuffer_ = std::make_unique<ControlTargetSingleBuffer<T, WAIST_JOINT_NUM * 2>>();
            armBuffer_[0] = std::make_unique<ControlTargetSingleBuffer<T, LEFT_ARM_JOINT_NUM * 2>>();
            armBuffer_[1] = std::make_unique<ControlTargetSingleBuffer<T, RIGHT_ARM_JOINT_NUM * 2>>();
            handBuffer_[0] = std::make_unique<ControlTargetSingleBuffer<T, LEFT_HAND_JOINT_NUM * 2>>();
            handBuffer_[1] = std::make_unique<ControlTargetSingleBuffer<T, RIGHT_HAND_JOINT_NUM * 2>>();
        }
    }

    void writeNewData(Robot_Control_Type type, const T *data, bool valid)
    {
        getBuffer_(type)->pushData(data, valid);
    }

    void writeNewData(Robot_Control_Type type, const std::vector<T> &data, bool valid)
    {
        getBuffer_(type)->pushData(data, valid);
    }

    const T *readData(Robot_Control_Type type)
    {
        return getBuffer_(type)->getDataPtr();
    }

    bool getDataValid(Robot_Control_Type type)
    {
        return getBuffer_(type)->getValid();
    }

    void switchBuffer(Robot_Control_Type type)
    {
        getBuffer_(type)->switchBuffer();
    }

    double getUpdateTime(Robot_Control_Type type)
    {
        return getBuffer_(type)->getUpdateTime();
    }

    void editCurrentData(Robot_Control_Type type, std::function<void(T const *)> editFunc)
    {
        getBuffer_(type)->editCurrentData(editFunc);
    }

  private:
    IControlTargetBufferBase<T> *getBuffer_(Robot_Control_Type type)
    {
        switch (type)
        {
        case Robot_Control_Type::Neck_Target:
            return neckBuffer_.get();
        case Robot_Control_Type::Waist_Target:
            return waistBuffer_.get();
        case Robot_Control_Type::Left_Arm_Target:
            return armBuffer_[0].get();
        case Robot_Control_Type::Right_Arm_Target:
            return armBuffer_[1].get();
        case Robot_Control_Type::Left_Hand_Target:
            return handBuffer_[0].get();
        case Robot_Control_Type::Right_Hand_Target:
            return handBuffer_[1].get();
        default:
            assert(false && "Invalid Robot_Control_Type");
            return nullptr;
        }
    }

    std::unique_ptr<IControlTargetBufferBase<T>> neckBuffer_;
    std::unique_ptr<IControlTargetBufferBase<T>> waistBuffer_;
    std::unique_ptr<IControlTargetBufferBase<T>> armBuffer_[2];
    std::unique_ptr<IControlTargetBufferBase<T>> handBuffer_[2];
};

#endif /* CONTROL_TARGET_BUFFER_HPP */
