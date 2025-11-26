#ifndef JOINT_ACCL_LIMIT_H
#define JOINT_ACCL_LIMIT_H

#include <vector>
#include <cmath>
#include <string>

class DynamicsState
{
  public:
    double position;
    double velocity;
    DynamicsState();
    DynamicsState(double position, double velocity);
    DynamicsState(const DynamicsState &other);
    virtual ~DynamicsState();
    DynamicsState clamp(const DynamicsState &state, double positionLim, double velocityLim);
    DynamicsState operator-(void) const;
    DynamicsState operator+(const DynamicsState &other) const;
    DynamicsState operator-(const DynamicsState &other) const;
    DynamicsState operator*(double scalar) const;
    DynamicsState operator/(double scalar) const;
    DynamicsState &operator=(const DynamicsState &state);
    std::string toStr(void) const;
    bool containsNaN(void) const;
};

class DynamicsProtecter1D
{
  public:
    DynamicsState lastOutput;

    DynamicsProtecter1D();
    DynamicsProtecter1D(double maxVelocity, double minVelocity, double maxAccl);
    virtual ~DynamicsProtecter1D();
    void initParams(double maxVelocity, double minVelocity, double maxAccl);
    template <typename T> DynamicsState calculateAcclClamp(const DynamicsState &currentState, T &&targetState, double time)
    {
        DynamicsState ts = std::forward<T>(targetState);
        return calculateAcclClampImpl_(currentState, ts, time);
    }

  private:
    double maxVelocity_;
    double minVelocity_;
    double maxAccl_;
    double maxAcclTell_;
    bool velocityLimitFlag_ = false;

    double clampVelocity_(double velocity) const;
    static DynamicsState calculateAcclStateCvt_(double time, double accl, const DynamicsState &startState);
    static DynamicsState integrateAcclToPosVel_(const std::vector<double> &timeList, const std::vector<double> &acclList,
                                                const DynamicsState &initState, double targetTime, const DynamicsState &overTimeDefault);
    std::pair<double, double> getCvtVelocityLim_(const DynamicsState &cvtFrame) const;
    DynamicsState handleZerosTargetZeroCurrent_(const DynamicsState &now, double minVel, double maxVel, double time);
    DynamicsState handleCurrentStateOverlimit_(const DynamicsState &now, double minVel, double maxVel, double time);
    DynamicsState handleZerosTargetPositiveCurrent_(const DynamicsState &now, double minVel, double maxVel, double time);
    DynamicsState handleZerosTargetPositiveNegativeState_(const DynamicsState &now, double minVel, double maxVel, double time);
    DynamicsState calculateAcclClampZerosTarget_(const DynamicsState &now, double minVel, double maxVel, double time);
    DynamicsState calculateAcclClampImpl_(const DynamicsState &currentState, DynamicsState &targetState, double time);
};

#endif /* JOINT_ACCL_LIMIT_H */
