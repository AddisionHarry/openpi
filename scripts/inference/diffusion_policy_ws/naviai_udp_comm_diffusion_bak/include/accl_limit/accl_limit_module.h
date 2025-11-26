#ifndef ACCL_LIMIT_MODULE
#define ACCL_LIMIT_MODULE

#include <cstddef>
#if defined USE_OSQP
#include "mpc_accl_limit.h"
#endif
#include "joint_accl_limit.h"

#include <Eigen/Dense>
#include <vector>

struct Limit
{
    double min;
    double max;

    Limit();
    explicit Limit(double limit);
    Limit(double min, double max);
    Limit(const std::vector<double> &limit);

    Limit operator*(double scalar) const;
    Limit operator/(double scalar) const;
    Limit &operator*=(double scalar);
    Limit &operator/=(double scalar);
    friend Limit operator*(double scalar, const Limit &limit);
};

struct Accl_Limit_Config
{
    double dt;
    Limit position;
    Limit speed;
    Limit acceleration;
    Limit jerk;
    Limit snap;

    Accl_Limit_Config(double dt = 0.0, double position_limit = 0.0, double speed_limit = 0.0, double acceleration_limit = 0.0,
                      double jerk_limit = 0.0, double snap_limit = 0.0);
    Accl_Limit_Config(double dt, const Limit &position_limit, double speed_limit, double acceleration_limit = 0.0, double jerk_limit = 0.0,
                      double snap_limit = 0.0);
    Accl_Limit_Config(double dt, const Limit &position_limit, const Limit &speed_limit, const Limit &acceleration_limit,
                      const Limit &jerk_limit = Limit(0.0, 0.0), const Limit &snap_limit = Limit(0.0, 0.0));

    Accl_Limit_Config operator*(double scalar) const;
    Accl_Limit_Config &operator*=(double scalar);
};

struct State
{
    Eigen::Vector4d state;

    State(double position = 0.0, double speed = 0.0, double acceleration = 0.0, double jerk = 0.0);
    State(const Eigen::Vector4d &state);

    double position(void) const;
    double speed(void) const;
    double acceleration(void) const;
    double jerk(void) const;

    void limitBounds(const Eigen::Vector4d &lb, const Eigen::Vector4d &ub);
    void limitBounds(const State &lb, const State &ub);

    friend std::ostream &operator<<(std::ostream &os, const State &obj);
};

class Accl_Limit_Smoother
{
  public:
    Accl_Limit_Smoother(const Accl_Limit_Config &config, const std::vector<State> &initialState, bool useMPC = true,
                        bool qpVerbose = false);
    Accl_Limit_Smoother(const std::vector<Accl_Limit_Config> &configs, const std::vector<State> &initialState, bool useMPC = true,
                        bool qpVerbose = false);
    ~Accl_Limit_Smoother() = default;

    const std::vector<Accl_Limit_Config> &readConfig(void) const;
    void setConfig(const std::vector<Accl_Limit_Config> &config);
    void setConfig(const Accl_Limit_Config &config, size_t index);

    const std::vector<State> &readCurrentState(void) const;
    void setCurrentState(const std::vector<State> &states);
    void setCurrentState(const State &state, size_t index);

    std::vector<Eigen::VectorXd> smooth(const std::vector<State> &target);
    const std::vector<State> &step(const std::vector<double> &control, double dt);
    const std::vector<State> &step(const Eigen::VectorXd &control, double dt);
    const std::vector<State> &step(size_t lastOutputIndex, double dt);

    static Eigen::MatrixXd mapSmoothedToMatrix(const std::vector<Eigen::VectorXd> &vec);

    std::vector<Eigen::VectorXd> lastOutput;

  private:
    bool useMPC_, qpVerbose_;
    size_t jointNum_;
    std::vector<State> currentState_, stateUpperBound_, stateLowerBound_;
    std::vector<Eigen::VectorXd> controlUpperBound_, controlLowerBound_;
    std::vector<Accl_Limit_Config> configs_;
    std::vector<DynamicsProtecter1D> acclSmoothers_;
#if defined USE_OSQP
    std::vector<LinearMPC_Smoother> mpcSmoothers_;
#endif
    std::vector<DynamicsState> acclLimitOutput_;
    size_t acclLimitOutputUsedCNT_;

    void initSmoother_(void);
    void updateConfig_(size_t index);
    void rangeStateLimits_(void);
};

#endif /* ACCL_LIMIT_MODULE */
