#if defined USE_OSQP

#include <chrono>
#include <cmath>
#include <cstddef>
#include <fstream>
#include <iostream>

#include "accl_limit/mpc_accl_limit.h"

#define TIME_STEP (1.0 / 50)
#define TIME_STEPS (1500)
#define PREDICT_HORIZON (150)
#define CONTROL_HORIZON (15)

#define Q_DIAG 37, 1, 0, 0
#define R_DIAG (2e-2)

class MPC_Loop_Test
{
  public:
    MPC_Loop_Test(const std::string &filename, size_t totalTestStep, double stepTime, const Eigen::MatrixXd &Q, const Eigen::MatrixXd &R,
                  bool useNormalized)
        : logger_(filename), smoother_(buildSmoother_(stepTime, Q, R, useNormalized)), totalTestStep_(totalTestStep), stepTime_(stepTime)

    {
        if (auto res = smoother_->initSolver(); res != LinearMPC_Smoother::SolverState::Success)
            std::cout << "Get fialed in initializing solver: " << smoother_->toString(res) << std::endl;
        x_.resize(4);
    }

    static Eigen::VectorXd getReference(double timeStamp)
    {
        Eigen::VectorXd res(4);
        if (timeStamp < 8.5)
            res << (1 + std::sin(0.7 * timeStamp)), 0.7 * std::cos(0.7 * timeStamp), 0, 0;
        else if (timeStamp < 12.5)
            res << 3, 0, 0, 0;
        else if (timeStamp < 17)
            res << (-3 + (timeStamp - 12.5) * 1.05), 1.05, 0, 0;
        else if (timeStamp < 21.5)
            res << -3, 0, 0, 0;
        else
            res << (-1 - std::cos(0.65 * (timeStamp - 22))), 0.65 * std::sin(0.65 * (timeStamp - 22)), 0, 0;
        return res;
    }

    double step(const Eigen::VectorXd &r, const Eigen::VectorXd &x)
    {
        Eigen::VectorXd res;
        if (auto state = smoother_->solve(x, r, res) != LinearMPC_Smoother::SolverState::Success)
            std::cout << "Get invalid solver state: " << state << std::endl;
        return res(0);
    }

    Eigen::VectorXd stepGetFullOutputSequence(const Eigen::VectorXd &r, const Eigen::VectorXd &x)
    {
        Eigen::VectorXd res;
        if (auto state = smoother_->solve(x, r, res); state != LinearMPC_Smoother::SolverState::Success)
            std::cout << "Get invalid solver state: " << smoother_->toString(state) << std::endl;
        return res;
    }

    void loop(const Eigen::VectorXd &x0, bool needLog = true)
    {
        x_ = x0;
        double u = 0, time = 0;
        for (size_t i = 0; i < totalTestStep_; ++i)
        {
            time = i * stepTime_;
            auto r = getReference(time);
            if (needLog)
            {
                auto start = std::chrono::high_resolution_clock::now();
                u = step(r, x_);
                std::chrono::duration<double, std::micro> duration_us = std::chrono::high_resolution_clock::now() - start;
                logger_.log(time, r, x_, u, duration_us.count());
            }
            else
                u = step(r, x_);
            x_ = smoother_->stepDynamics(x_, u);
        }
    }

    Eigen::VectorXd stepDynamics(const Eigen::VectorXd &x0, double u)
    {
        return smoother_->stepDynamics(x0, u);
    }

  private:
    class CSVLogger
    {
      public:
        CSVLogger(const std::string &filename)
        {
            file_.open(filename);
            if (!file_.is_open())
                throw std::runtime_error("Failed to open CSV file: " + filename);
            file_ << "t,rp,rv,ra,rj,p,v,a,j,s,solveTime_us\n";
        }

        ~CSVLogger()
        {
            if (file_.is_open())
                file_.close();
        }

        void log(double t, const Eigen::VectorXd &r, const Eigen::VectorXd &x, double u, double solveTime)
        {
            if (!file_.is_open())
            {
                std::cerr << "CSV file is not open for writing!" << std::endl;
                return;
            }
            file_ << t << "," << r(0) << "," << r(1) << "," << r(2) << "," << r(3) << "," << x(0) << "," << x(1) << "," << x(2) << ","
                  << x(3) << "," << u << "," << solveTime << "\n";
        }

      private:
        std::ofstream file_;
    };

    std::unique_ptr<LinearMPC_Smoother> buildSmoother_(double stepTime, const Eigen::MatrixXd &Q, const Eigen::MatrixXd &R,
                                                       bool useNormalized)
    {
        Eigen::VectorXd ulim(1), xlim(4);
        ulim << 200;
        xlim << 3, 2.8, 4.6, 50;
        if (useNormalized)
            return std::make_unique<NormalizedLinearMPC_Smoother>(
                NormalizedLinearChainIntegrator(stepTime, xlim.array().inverse(), ulim.array().inverse()), PREDICT_HORIZON, CONTROL_HORIZON,
                Q, R, xlim, -xlim, ulim, -ulim);
        else
            return std::make_unique<LinearMPC_Smoother>(
                LinearMPC_Smoother(LinearChainIntegrator(stepTime), PREDICT_HORIZON, CONTROL_HORIZON, Q, R, xlim, -xlim, ulim, -ulim));
    }

    CSVLogger logger_;
    std::unique_ptr<LinearMPC_Smoother> smoother_;
    Eigen::VectorXd x_;
    size_t totalTestStep_;
    double stepTime_;
};

void testSingleOptimize(const Eigen::VectorXd &x0, const Eigen::VectorXd &r)
{
    Eigen::Vector4d Q_diag_elements(Q_DIAG);

    {
        Eigen::VectorXd ulim(1), xlim(4);
        ulim << 200;
        xlim << 3, 2.8, 4.6, 50;
        auto smoother = LinearMPC_Smoother(LinearChainIntegrator(TIME_STEP), PREDICT_HORIZON, CONTROL_HORIZON, Q_diag_elements.asDiagonal(),
                                           R_DIAG * Eigen::MatrixXd::Identity(1, 1), xlim, -xlim, ulim, -ulim);
        if (auto res = smoother.initSolver(true); res != LinearMPC_Smoother::SolverState::Success)
            std::cout << "Get fialed in initializing solver: " << smoother.toString(res) << std::endl;
        Eigen::VectorXd res;
        if (auto state = smoother.solve(x0, r, res); state != LinearMPC_Smoother::SolverState::Success)
            std::cout << "Get invalid solver state: " << smoother.toString(state) << std::endl;

        std::cout << "x: " << x0.transpose() << std::endl;
        std::cout << "Given refernce: " << r.transpose() << std::endl;
        std::cout << "Get planned result: " << res.transpose() << std::endl;
    }

    {
        std::cout << "***************" << std::endl;
        auto test = MPC_Loop_Test("/root/TeleVision-ThreeJS/logs/test_mpc.txt", TIME_STEPS, TIME_STEP, Q_diag_elements.asDiagonal(),
                                  R_DIAG * Eigen::MatrixXd::Identity(1, 1), false);
        std::cout << "x: " << x0.transpose() << std::endl;
        std::cout << "Given refernce: " << test.getReference(TIME_STEP).transpose() << std::endl;
        auto u = test.stepGetFullOutputSequence(test.getReference(0), x0);
        std::cout << "Get planned result: " << u.transpose() << std::endl;
        std::cout << "Stepped x: " << test.stepDynamics(x0, u(0)).transpose() << std::endl;
    }

    {
        std::cout << "***************" << std::endl;
        Eigen::VectorXd ulim(1), xlim(4);
        ulim << 200;
        xlim << 3, 2.8, 4.6, 50;
        auto test = JointLinearMPC_Smoother(LinearChainIntegrator(TIME_STEP), PREDICT_HORIZON, CONTROL_HORIZON,
                                            Q_diag_elements.asDiagonal(), R_DIAG * Eigen::MatrixXd::Identity(1, 1),
                                            std::vector<Eigen::MatrixXd>(7, xlim), std::vector<Eigen::MatrixXd>(7, -xlim),
                                            std::vector<Eigen::MatrixXd>(7, ulim), std::vector<Eigen::MatrixXd>(7, -ulim));
        test.initSolver(true);
        std::vector<Eigen::VectorXd> res;
        if (auto state = test.solve(std::vector<Eigen::VectorXd>(7, x0), std::vector<Eigen::VectorXd>(7, r), res);
            state != LinearMPC_Smoother::SolverState::Success)
            std::cout << "Get invalid solver state: " << LinearMPC_Smoother::toString(state) << std::endl;
        std::cout << "Get result:" << std::endl;
        for (size_t i = 0; i < res.size(); ++i)
            std::cout << res.at(i).transpose() << std::endl;
    }
}

void testLoopMPC(const Eigen::VectorXd &x0)
{
    Eigen::Vector4d Q_diag_elements(Q_DIAG);
    auto test = MPC_Loop_Test("/root/TeleVision-ThreeJS/logs/test_mpc.txt", TIME_STEPS, TIME_STEP, Q_diag_elements.asDiagonal(),
                              R_DIAG * Eigen::MatrixXd::Identity(1, 1), false);
    test.loop(x0);
    auto start = std::chrono::high_resolution_clock::now();
    test.loop(x0, false);
    std::chrono::duration<double, std::micro> duration_us = std::chrono::high_resolution_clock::now() - start;
    std::cout << "Total MPC time: " << duration_us.count() << " us" << std::endl;
    std::cout << "Average per step: " << duration_us.count() / TIME_STEPS << " us/iteration" << std::endl;
}
#endif

int main(int argc, char **argv)
{
#if defined(USE_OSQP)
#define TEST_SINGLE
// #undef TEST_SINGLE
#if defined TEST_SINGLE
    Eigen::VectorXd x0(4), r(4);
    x0 << -2.03329, -1.99529, 0.255555, -4.33337;
    r << 1.01333, 0.799929, 0, 0;
    // x0 << -2, -2, 0.3, -1;
    // r << 1, 0.8, 0, 0;
    testSingleOptimize(x0, r);
#else
    Eigen::VectorXd x0(4);
    x0 << -2, -2, 0.3, -1;
    testLoopMPC(x0);
#endif
#endif
    return 0;
}
