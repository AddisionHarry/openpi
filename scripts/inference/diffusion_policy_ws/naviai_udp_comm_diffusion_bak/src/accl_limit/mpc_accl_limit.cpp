#if defined USE_OSQP

#include "accl_limit/mpc_accl_limit.h"

#include <functional>
#include <memory>
#include <stdexcept>
#include <unsupported/Eigen/KroneckerProduct>

#define MPC_INTRGRATE_ORDER 4

LinearChainIntegrator::LinearChainIntegrator(double dt_, bool inputInverse) : dt(dt_)
{
    double dt2 = dt * dt;
    double dt3 = dt2 * dt;
#if ((defined MPC_TEST_THIRD_INTRGRATE) && (MPC_TEST_THIRD_INTRGRATE == 2))
    // clang-format off
    A_ << 1, dt, 0, 0,
          0,  1, 0, 0,
          0,  0, 0, 0,
          0,  0, 0, 0;
    B_ << dt2 / 2.0, dt, 0, 0;
    // clang-format on
#elif ((defined MPC_TEST_THIRD_INTRGRATE) && (MPC_TEST_THIRD_INTRGRATE == 3))
    // clang-format off
    A_ << 1, dt, 0.5 * dt2, 0,
          0,  1,        dt, 0,
          0,  0,         1, 0,
          0,  0,         0, 0;
    B_ << dt3 / 6.0, dt2 / 2.0, dt, 0;
    // clang-format on
#elif ((defined MPC_TEST_THIRD_INTRGRATE) && (MPC_TEST_THIRD_INTRGRATE == 4)) || (!(defined MPC_TEST_THIRD_INTRGRATE))
    double dt4 = dt3 * dt;
    // clang-format off
    A_ << 1, dt, 0.5 * dt2, (1.0 / 6.0) * dt3,
          0,  1,        dt,         0.5 * dt2,
          0,  0,         1,                dt,
          0,  0,         0,                 1;
    B_ << dt4 / 24.0, dt3 / 6.0, dt2 / 2.0, dt;
    // clang-format on
#else
#error "Unsupported MPC integration order, please define MPC_TEST_THIRD_INTRGRATE to 2, 3 or 4."
#endif
    B_ *= inputInverse ? -1.0 : 1.0;
}

LinearChainIntegrator::~LinearChainIntegrator()
{
}

LinearChainIntegrator::LinearChainIntegrator(const Eigen::Matrix4d &A, const Eigen::Vector4d &B) : dt(std::abs(B(3))), A_(A), B_(B)
{
}

LinearChainIntegrator::LinearChainIntegrator(const LinearChainIntegrator &integrator, bool inputInverse)
{
    dt = integrator.dt;
    A_ = integrator.A();
    B_ = (inputInverse ? -1.0 : 1.0) * integrator.B();
}

Eigen::Vector4d LinearChainIntegrator::step(const Eigen::Vector4d &x, double u) const
{
    return A_ * x + B_ * u;
}

const Eigen::Matrix4d &LinearChainIntegrator::A(void) const
{
    return A_;
}

const Eigen::Vector4d &LinearChainIntegrator::B(void) const
{
    return B_;
}

void LinearChainIntegrator::setInputInverse(void)
{
    B_ *= -1;
}

LinearMPC_Smoother::LinearMPC_Smoother(const LinearChainIntegrator &dynamics, int N, int Nu, const Eigen::MatrixXd &Q,
                                       const Eigen::MatrixXd &R, double controlSmoothRate, const Eigen::MatrixXd &xmax,
                                       const Eigen::MatrixXd &xmin, const Eigen::MatrixXd &umax, const Eigen::MatrixXd &umin)
    : dynamics_(std::make_unique<LinearChainIntegrator>(dynamics, true)), controlSmoothRate_(controlSmoothRate), N_(N), Nu_(Nu), Q_(Q),
      R_(R), umin_(umin), umax_(umax), xmin_(xmin), xmax_(xmax), inited_(false), initing_(false), refValid_(false)
{
    buildMatrices_();
}

LinearMPC_Smoother::LinearMPC_Smoother(std::unique_ptr<LinearChainIntegrator> dynamics, int N, int Nu, const Eigen::MatrixXd &Q,
                                       const Eigen::MatrixXd &R, double controlSmoothRate, const Eigen::MatrixXd &xmax,
                                       const Eigen::MatrixXd &xmin, const Eigen::MatrixXd &umax, const Eigen::MatrixXd &umin)
    : dynamics_(std::move(dynamics)), controlSmoothRate_(controlSmoothRate), N_(N), Nu_(Nu), Q_(Q), R_(R), umin_(umin), umax_(umax),
      xmin_(xmin), xmax_(xmax), inited_(false), initing_(false), refValid_(false)
{
    dynamics_->setInputInverse();
    buildMatrices_();
}

LinearMPC_Smoother::~LinearMPC_Smoother()
{
}

void LinearMPC_Smoother::buildMatrices_(void)
{
    buildCondensedMatrices_();
    buildCostMatrices_();
    buildConstraintMatrices_();
    P_ = tildeR_ + Phi_.transpose() * tildeQ_ * Phi_;
    qLinear_ = Phi_.transpose() * tildeQ_.transpose() * Tau_;
}

const char *LinearMPC_Smoother::toString(SolverState state)
{
    switch (state)
    {
    case SolverState::Success:
        return "Success";
    case SolverState::UpdateBoundFailure:
        return "UpdateBoundFailure";
    case SolverState::UpdateGradientFailure:
        return "UpdateGradientFailure";
    case SolverState::SolverFailure:
        return "SolverFailure";
    case SolverState::InitHessianMatrixFalure:
        return "InitHessianMatrixFalure";
    case SolverState::InitLinearConstraintsMatrixFalure:
        return "InitLinearConstraintsMatrixFalure";
    case SolverState::InitSolverFalure:
        return "InitSolverFalure";
    default:
        return "Unknown";
    }
}

LinearMPC_Smoother::SolverState LinearMPC_Smoother::initSolver(bool isVerbose)
{
    initing_ = true;
    solver_ = std::make_unique<OsqpEigen::Solver>();
    // solver_->settings()->setWarmStart(false);
    solver_->settings()->setWarmStart(true);
    solver_->settings()->setMaxIteration(120);
    solver_->settings()->setVerbosity(isVerbose);
    solver_->settings()->setScaling(3);
    solver_->settings()->setRho(0.08);
    solver_->settings()->setAbsoluteTolerance(4e-4);
    solver_->settings()->setRelativeTolerance(1e-3);
    solver_->settings()->setPolish(false);
    solver_->settings()->setCheckTermination(true);
    solver_->data()->setNumberOfVariables(Nu_);
    solver_->data()->setNumberOfConstraints(dynamics_->stateDim * N_ + dynamics_->inputDim * Nu_);
    if (!solver_->data()->setHessianMatrix(P_))
        return SolverState::InitHessianMatrixFalure;
    if (!updateOSQPGradient_(Eigen::Vector4d(1, 0, 0, 0)))
        return SolverState::UpdateGradientFailure;
    if (!solver_->data()->setLinearConstraintsMatrix(A_))
        return SolverState::InitLinearConstraintsMatrixFalure;
    if (!updateOSQPBounds_(Eigen::Vector4d(1, 0, 0, 0), Eigen::Vector4d(0, 0, 0, 0)))
        return SolverState::UpdateBoundFailure;
    if (!solver_->initSolver())
        return SolverState::InitSolverFalure;
    inited_ = true;
    return SolverState::Success;
}

void LinearMPC_Smoother::setBounds(const Eigen::VectorXd &umin, const Eigen::VectorXd &umax, const Eigen::VectorXd &xmin,
                                   const Eigen::VectorXd &xmax)
{
    assert(umin.size() > 0);
    assert(umin.size() == umax.size());
    assert(umin.size() == umin_.size());
    assert(xmin.size() > 0);
    assert(xmin.size() == xmax.size());
    assert(xmin.size() == xmin_.size());
    umin_ = umin;
    umax_ = umax;
    xmin_ = xmin;
    xmax_ = xmax;
}

void LinearMPC_Smoother::buildCondensedMatrices_(void)
{
    std::vector<Triplet> triplets_Tau;
    const Eigen::Matrix4d &A = dynamics_->A();

    std::vector<Eigen::Matrix4d> A_powers(N_ + 1);
    A_powers[0] = Eigen::Matrix4d::Identity();
    for (int i = 1; i <= N_; ++i)
        A_powers[i] = A * A_powers[i - 1];

    for (int i = 0; i < N_; ++i)
    {
        const Eigen::Matrix4d &A_i = A_powers[i + 1]; // A^{i+1}
        for (int row = 0; row < dynamics_->stateDim; ++row)
            for (int col = 0; col < dynamics_->stateDim; ++col)
                triplets_Tau.emplace_back(i * dynamics_->stateDim + row, col, A_i(row, col));
    }
    Tau_.resize(dynamics_->stateDim * N_, dynamics_->stateDim); // 4N * 4
    Tau_.setFromTriplets(triplets_Tau.begin(), triplets_Tau.end());

    std::vector<Triplet> triplets_Phi;
    const Eigen::Vector4d &B = dynamics_->B();
    Eigen::Vector4d sumPrev = Eigen::Vector4d::Zero();
    for (int i = 0; i < N_; ++i)
    {
        if (i < Nu_)
        {
            for (int j = 0; j <= i; ++j)
            {
                Eigen::Vector4d A_power_B = A_powers[i - j] * B;
                for (int row = 0; row < dynamics_->stateDim; ++row)
                    triplets_Phi.emplace_back(i * dynamics_->stateDim + row, j, A_power_B(row));
            }
        }
        else
        {
            for (int j = 0; j < Nu_ - 1; ++j)
            {
                Eigen::Vector4d A_power_B = A_powers[i - j] * B;
                for (int row = 0; row < dynamics_->stateDim; ++row)
                    triplets_Phi.emplace_back(i * dynamics_->stateDim + row, j, A_power_B(row));
            }
            sumPrev += A_powers[i - Nu_ + 1] * B;
            for (int row = 0; row < dynamics_->stateDim; ++row)
                triplets_Phi.emplace_back(i * dynamics_->stateDim + row, Nu_ - 1, sumPrev(row));
        }
    }
    Phi_.resize(dynamics_->stateDim * N_, Nu_); // 4N * Nu
    Phi_.setFromTriplets(triplets_Phi.begin(), triplets_Phi.end());
}

void LinearMPC_Smoother::buildCostMatrices_(void)
{
    std::vector<Triplet> triplets_Q;
    for (int row = 0; row < dynamics_->stateDim; ++row)
    {
        for (int col = 0; col < dynamics_->stateDim; ++col)
        {
            double val = Q_(row, col);
            if (val != 0.0)
            {
                for (int i = 0; i < N_; ++i)
                    triplets_Q.emplace_back(i * dynamics_->stateDim + row, i * dynamics_->stateDim + col,
                                            (i == N_ - 1) ? val * (N_ - Nu_ + 1) : val);
            }
        }
    }
    tildeQ_.resize(dynamics_->stateDim * N_, dynamics_->stateDim * N_);
    tildeQ_.setFromTriplets(triplets_Q.begin(), triplets_Q.end());

    std::vector<Triplet> triplets_R;
    for (int row = 0; row < dynamics_->inputDim; ++row)
    {
        for (int col = 0; col < dynamics_->inputDim; ++col)
        {
            double val = R_(row, col);
            if (val != 0.0)
                for (int i = 0; i < Nu_; ++i)
                    triplets_R.emplace_back(i * dynamics_->inputDim + row, i * dynamics_->inputDim + col, val);
        }
    }
    tildeR_.resize(dynamics_->inputDim * Nu_, dynamics_->inputDim * Nu_);
    tildeR_.setFromTriplets(triplets_R.begin(), triplets_R.end());

    std::vector<Eigen::Triplet<double>> triplets_smooth;
    for (int i = 0; i < Nu_; ++i)
    {
        for (int inputIdx = 0; inputIdx < dynamics_->inputDim; ++inputIdx)
        {
            int row = i * dynamics_->inputDim + inputIdx;
            int col = i * dynamics_->inputDim + inputIdx;
            if (i == 0 || i == Nu_ - 1)
                triplets_smooth.emplace_back(row, col, controlSmoothRate_ * 1.0);
            else
                triplets_smooth.emplace_back(row, col, controlSmoothRate_ * 2.0);
            if (i > 0)
            {
                int col_prev = (i - 1) * dynamics_->inputDim + inputIdx;
                triplets_smooth.emplace_back(row, col_prev, controlSmoothRate_ * -1.0);
            }
            if (i < Nu_ - 1)
            {
                int col_next = (i + 1) * dynamics_->inputDim + inputIdx;
                triplets_smooth.emplace_back(row, col_next, controlSmoothRate_ * -1.0);
            }
        }
    }
    SparseMatrixXd smoothMat(tildeR_.rows(), tildeR_.cols());
    smoothMat.setFromTriplets(triplets_smooth.begin(), triplets_smooth.end());
    tildeR_ += smoothMat;
}

void LinearMPC_Smoother::buildConstraintMatrices_(void)
{
    Eigen::SparseMatrix<double> I(Nu_, Nu_);
    std::vector<Eigen::Triplet<double>> identityTriplets;
    for (int i = 0; i < Nu_; ++i)
        identityTriplets.emplace_back(i, i, 1.0);
    I.setFromTriplets(identityTriplets.begin(), identityTriplets.end());

    Eigen::SparseMatrix<double> stacked(4 * N_ + Nu_, Nu_);
    stacked.reserve(Phi_.nonZeros() + I.nonZeros());
    for (int col = 0; col < Phi_.outerSize(); ++col)
        for (Eigen::SparseMatrix<double>::InnerIterator it(Phi_, col); it; ++it)
            stacked.insert(it.row(), it.col()) = it.value();
    for (int col = 0; col < I.outerSize(); ++col)
        for (Eigen::SparseMatrix<double>::InnerIterator it(I, col); it; ++it)
            stacked.insert(it.row() + 4 * N_, it.col()) = it.value();
    stacked.makeCompressed();
    A_ = stacked;
}

Eigen::VectorXd LinearMPC_Smoother::buildReferenceVector_(const Eigen::VectorXd &r)
{
    if (r.size() != dynamics_->stateDim)
        throw std::runtime_error("Input r must be of size 4.");
    Eigen::Vector4d x = Eigen::Map<const Eigen::Vector4d>(r.data());
    if (refValid_ && (x.array() == cachedR_.array()).all())
        return cachedRef_;
    Eigen::VectorXd ref(N_ * 4);
    for (int i = 0; i < N_; ++i)
    {
        ref.segment<4>(i * 4) = x;
        x = dynamics_->step(x, 0.0);
    }
    cachedR_ = Eigen::Map<const Eigen::Vector4d>(r.data());
    cachedRef_ = ref;
    refValid_ = true;
    return ref;
}

bool LinearMPC_Smoother::updateOSQPGradient_(const Eigen::VectorXd &e)
{
    q_ = qLinear_ * e;
    if (inited_)
        return static_cast<bool>(solver_->updateGradient(q_));
    else if (initing_)
        return static_cast<bool>(solver_->data()->setGradient(q_));
    else
        return true;
}

bool LinearMPC_Smoother::updateOSQPBounds_(const Eigen::VectorXd &x0, const Eigen::VectorXd &r)
{
    assert(xmin_.size() == dynamics_->stateDim);
    assert(xmax_.size() == dynamics_->stateDim);
    assert(umin_.size() == dynamics_->inputDim);
    assert(umax_.size() == dynamics_->inputDim);
    Eigen::VectorXd refTraj = buildReferenceVector_(r) - Tau_ * (r - x0);
    assert(refTraj.size() == dynamics_->stateDim * N_);
    lowerBound_.resize(dynamics_->stateDim * N_ + dynamics_->inputDim * Nu_);
    upperBound_.resize(dynamics_->stateDim * N_ + dynamics_->inputDim * Nu_);
    lowerBound_.setZero();
    upperBound_.setZero();
    lowerBound_.head(dynamics_->stateDim * N_) = (refTraj.array() - xmax_.replicate(N_, 1).array()).matrix();
    lowerBound_.tail(dynamics_->inputDim * Nu_) = umin_.replicate(Nu_, 1);
    upperBound_.head(dynamics_->stateDim * N_) = (refTraj.array() - xmin_.replicate(N_, 1).array()).matrix();
    upperBound_.tail(dynamics_->inputDim * Nu_) = umax_.replicate(Nu_, 1);
    if (inited_)
        return static_cast<bool>(solver_->updateBounds(lowerBound_, upperBound_));
    else if (initing_)
        return static_cast<bool>(solver_->data()->setBounds(lowerBound_, upperBound_));
    else
        return true;
}

LinearMPC_Smoother::SolverState LinearMPC_Smoother::updateSolver(const Eigen::VectorXd &x0, const Eigen::VectorXd &r)
{
    Eigen::VectorXd error = r - x0;
    if (!updateOSQPGradient_(r - x0))
        return SolverState::UpdateGradientFailure;
    if (!updateOSQPBounds_(x0, r))
        return SolverState::UpdateBoundFailure;
    return SolverState::Success;
}

LinearMPC_Smoother::SolverState LinearMPC_Smoother::solve(const Eigen::VectorXd &x0, const Eigen::VectorXd &r, Eigen::VectorXd &uSeqOutput)
{
    if (auto res = updateSolver(x0, r); res != SolverState::Success)
        return res;
    if (solver_->solveProblem() != OsqpEigen::ErrorExitFlag::NoError)
        return SolverState::SolverFailure;
    uSeqOutput = solver_->getSolution().cast<double>();
    return SolverState::Success;
}

Eigen::VectorXd LinearMPC_Smoother::limitOutput(const Eigen::VectorXd &uSeqOutput)
{
    return uSeqOutput.cwiseMin(umax_.replicate(Nu_, 1)).cwiseMax(umin_.replicate(Nu_, 1));
}

Eigen::VectorXd LinearMPC_Smoother::stepDynamics(const Eigen::VectorXd &x0, double u)
{
    return dynamics_->step(x0, -u);
}

Eigen::VectorXd LinearMPC_Smoother::stepDynamics(const Eigen::VectorXd &x0, double u, double dt)
{
    if (std::abs(dt - dynamics_->dt) > 1e-5)
        return LinearChainIntegrator(dt, false).step(x0, -u);
    else
        return stepDynamics(x0, -u);
}

double LinearMPC_Smoother::getCurrentMPCCostValue(const Eigen::VectorXd &uSeqOutput) const
{
    assert(uSeqOutput.size() == dynamics_->inputDim * Nu_);
    return (uSeqOutput.transpose() * P_ * uSeqOutput + q_.transpose() * uSeqOutput)(0);
}

Eigen::VectorXd LinearMPC_Smoother::getUnconstrainedSolution(void) const
{
    Eigen::SparseLU<Eigen::SparseMatrix<double>> solver;
    solver.analyzePattern(P_);
    solver.factorize(P_);
    return solver.solve(-q_);
}

Eigen::MatrixXd LinearMPC_Smoother::getHessianMatrix(void) const
{
    return Eigen::MatrixXd(P_);
}

const Eigen::SparseMatrix<double> &LinearMPC_Smoother::getHessianMatrixSparsed(void) const
{
    return P_;
}

Eigen::MatrixXd LinearMPC_Smoother::getPhiMatrix(void) const
{
    return Eigen::MatrixXd(Phi_);
}

const Eigen::VectorXd &LinearMPC_Smoother::getGradientVector(void) const
{
    return q_;
}

const Eigen::VectorXd &LinearMPC_Smoother::getBound(bool isLower) const
{
    return isLower ? lowerBound_ : upperBound_;
}

Eigen::MatrixXd LinearMPC_Smoother::getLinearConstraintMatrix(void) const
{
    return Eigen::MatrixXd(A_);
}

const Eigen::SparseMatrix<double> &LinearMPC_Smoother::getLinearConstraintMatrixSparsed(void) const
{
    return A_;
}

NormalizedLinearChainIntegrator::NormalizedLinearChainIntegrator(double dt, const Eigen::VectorXd &xscale, const Eigen::VectorXd &uscale,
                                                                 bool inputInverse)
    : LinearChainIntegrator(dt, inputInverse)
{
    xScale_ = xscale.asDiagonal();
    uScale_ = uscale.asDiagonal();
    tildeA_ = xScale_.inverse() * LinearChainIntegrator::A() * xScale_;
    tildeB_ = xScale_.inverse() * LinearChainIntegrator::B() * uScale_;
}

NormalizedLinearChainIntegrator::NormalizedLinearChainIntegrator(const NormalizedLinearChainIntegrator &integrator, bool inputInverse)
    : LinearChainIntegrator(integrator, inputInverse), xScale_(integrator.xScale_), uScale_(integrator.uScale_),
      tildeA_(integrator.tildeA_), tildeB_(integrator.tildeB_)
{
}

NormalizedLinearChainIntegrator::~NormalizedLinearChainIntegrator()
{
}

Eigen::Vector4d NormalizedLinearChainIntegrator::step(const Eigen::Vector4d &x, double u) const
{
    return tildeA_ * x + tildeB_ * u;
}

Eigen::Vector4d NormalizedLinearChainIntegrator::stepOutter(const Eigen::Vector4d &x, double u) const
{
    return LinearChainIntegrator::step(x, u);
}

const Eigen::Matrix4d &NormalizedLinearChainIntegrator::A(void) const
{
    return tildeA_;
}

const Eigen::Vector4d &NormalizedLinearChainIntegrator::B(void) const
{
    return tildeB_;
}

const Eigen::Matrix4d &NormalizedLinearChainIntegrator::AOutter(void) const
{
    return LinearChainIntegrator::A();
}

const Eigen::Vector4d &NormalizedLinearChainIntegrator::BOutter(void) const
{
    return LinearChainIntegrator::B();
}

const Eigen::MatrixXd &NormalizedLinearChainIntegrator::getXScale(void) const
{
    return xScale_;
}

const Eigen::MatrixXd &NormalizedLinearChainIntegrator::getUScale(void) const
{
    return uScale_;
}

NormalizedLinearMPC_Smoother::NormalizedLinearMPC_Smoother(const NormalizedLinearChainIntegrator &dynamics, int N, int Nu,
                                                           const Eigen::MatrixXd &Q, const Eigen::MatrixXd &R, const Eigen::MatrixXd &xmax,
                                                           const Eigen::MatrixXd &xmin, const Eigen::MatrixXd &umax,
                                                           const Eigen::MatrixXd &umin)
    : LinearMPC_Smoother(std::make_unique<NormalizedLinearChainIntegrator>(dynamics, false), N, Nu,
                         dynamics.getXScale().transpose() * Q * dynamics.getXScale(),
                         dynamics.getUScale().transpose() * R * dynamics.getUScale(), 0.0, dynamics.getXScale() * xmax,
                         dynamics.getXScale() * xmin, dynamics.getUScale() * umax, dynamics.getUScale() * umin),
      transformUSeqMat_(Eigen::kroneckerProduct(Eigen::MatrixXd::Identity(Nu, Nu), dynamics.getUScale().inverse()))
{
    initSolver();
    solver_->settings()->resetDefaultSettings();
}

NormalizedLinearMPC_Smoother::~NormalizedLinearMPC_Smoother()
{
}

LinearMPC_Smoother::SolverState NormalizedLinearMPC_Smoother::solve(const Eigen::VectorXd &x0, const Eigen::VectorXd &r,
                                                                    Eigen::VectorXd &uSeqOutput)
{
    if (auto ptr = dynamic_cast<NormalizedLinearChainIntegrator *>(dynamics_.get()))
    {
        auto state = LinearMPC_Smoother::solve(ptr->getXScale() * x0, ptr->getXScale() * r, uSeqOutput);
        if (state != SolverState::Success)
            return state;
        uSeqOutput = transformUSeqMat_ * uSeqOutput;
        return SolverState::Success;
    }
    std::cerr << "Cast failed\n";
    return SolverState::InnerGrammarFailure;
}

Eigen::VectorXd NormalizedLinearMPC_Smoother::stepDynamics(const Eigen::VectorXd &x0, double u)
{
    if (auto ptr = dynamic_cast<NormalizedLinearChainIntegrator *>(dynamics_.get()))
        return ptr->stepOutter(x0, u);
    std::cerr << "Cast failed\n";
    return Eigen::VectorXd();
}

JointLinearMPC_Smoother::JointLinearMPC_Smoother(const LinearChainIntegrator &dynamics, int N, int Nu, const Eigen::MatrixXd &Q,
                                                 const Eigen::MatrixXd &R, const std::vector<Eigen::MatrixXd> &xmax,
                                                 const std::vector<Eigen::MatrixXd> &xmin, const std::vector<Eigen::MatrixXd> &umax,
                                                 const std::vector<Eigen::MatrixXd> &umin)
    : inited_(false), N_(N), Nu_(Nu), numAgents_(xmax.size())
{
    assert(xmax.size() == xmin.size());
    assert(xmax.size() == umax.size());
    assert(xmax.size() == umin.size());
    hessians_.reserve(xmax.size());
    constraints_.reserve(xmax.size());
    upperBounds_.reserve(xmax.size());
    lowerBounds_.reserve(xmax.size());
    gradients_.reserve(xmax.size());
    for (size_t i = 0; i < xmax.size(); ++i)
    {
        smoothers_.emplace_back(
            std::make_unique<LinearMPC_Smoother>(dynamics, N, Nu, Q, R, 0.0, xmax.at(i), xmin.at(i), umax.at(i), umin.at(i)));
        smoothers_.at(i)->updateSolver(Eigen::Vector4d(1, 0, 0, 0), Eigen::Vector4d(0, 0, 0, 0));
        hessians_.emplace_back(smoothers_.at(i)->getHessianMatrixSparsed());
        constraints_.emplace_back(smoothers_.at(i)->getLinearConstraintMatrixSparsed());
        upperBounds_.emplace_back(smoothers_.at(i)->getBound(false));
        lowerBounds_.emplace_back(smoothers_.at(i)->getBound(true));
        gradients_.emplace_back(smoothers_.at(i)->getGradientVector());
    }
    bigP_ = blockDiagonalConcat_(hessians_);
    bigA_ = blockDiagonalConcat_(constraints_);
    bigq_ = concatVectors_(gradients_);
    biglb_ = concatVectors_(lowerBounds_);
    bigub_ = concatVectors_(upperBounds_);
}

JointLinearMPC_Smoother::~JointLinearMPC_Smoother()
{
}

LinearMPC_Smoother::SolverState JointLinearMPC_Smoother::initSolver(bool isVerbose)
{
    solver_.settings()->setWarmStart(true);
    solver_.settings()->setMaxIteration(200 * numAgents_);
    solver_.settings()->setVerbosity(isVerbose);
    solver_.settings()->setScaling(2);
    solver_.settings()->setAbsoluteTolerance(1e-5);
    solver_.settings()->setRelativeTolerance(1e-5);
    solver_.data()->setNumberOfVariables(Nu_ * numAgents_);
    solver_.data()->setNumberOfConstraints((LinearChainIntegrator::stateDim * N_ + LinearChainIntegrator::inputDim * Nu_) * numAgents_);
    if (!solver_.data()->setHessianMatrix(bigP_))
        return SolverState::InitHessianMatrixFalure;
    if (!solver_.data()->setGradient(bigq_))
        return SolverState::UpdateGradientFailure;
    if (!solver_.data()->setLinearConstraintsMatrix(bigA_))
        return SolverState::InitLinearConstraintsMatrixFalure;
    if (!solver_.data()->setBounds(biglb_, bigub_))
        return SolverState::UpdateBoundFailure;
    if (!solver_.initSolver())
        return SolverState::InitSolverFalure;
    inited_ = true;
    return SolverState::Success;
}

LinearMPC_Smoother::SolverState JointLinearMPC_Smoother::updateSolver_(const std::vector<Eigen::VectorXd> &x0List,
                                                                       const std::vector<Eigen::VectorXd> &rList)
{
    for (int i = 0; i < numAgents_; ++i)
    {
        auto &sm = *smoothers_.at(i);
        if (auto st = sm.updateSolver(x0List.at(i), rList.at(i)); st != SolverState::Success)
            return st;
        gradients_.at(i) = sm.getGradientVector();
        lowerBounds_.at(i) = sm.getBound(true);
        upperBounds_.at(i) = sm.getBound(false);
    }
    bigq_ = concatVectors_(gradients_);
    biglb_ = concatVectors_(lowerBounds_);
    bigub_ = concatVectors_(upperBounds_);

    if (!inited_)
        return SolverState::NotInited;
    if (!solver_.data()->setGradient(bigq_))
        return SolverState::UpdateGradientFailure;
    if (!solver_.data()->setBounds(biglb_, bigub_))
        return SolverState::UpdateBoundFailure;
    return SolverState::Success;
}

LinearMPC_Smoother::SolverState JointLinearMPC_Smoother::solve(const std::vector<Eigen::VectorXd> &x0List,
                                                               const std::vector<Eigen::VectorXd> &rList,
                                                               std::vector<Eigen::VectorXd> &uSeqOutputs)
{
    if (auto res = updateSolver_(x0List, rList); res != SolverState::Success)
        return res;
    if (!inited_)
        return SolverState::NotInited;
    if (solver_.solveProblem() != OsqpEigen::ErrorExitFlag::NoError)
        return SolverState::SolverFailure;
    auto xOpt = solver_.getSolution().cast<double>();
    uSeqOutputs.resize(numAgents_);
    for (int i = 0; i < numAgents_; ++i)
        uSeqOutputs[i] = xOpt.segment(Nu_ * LinearChainIntegrator::inputDim * i, Nu_ * LinearChainIntegrator::inputDim);
    return SolverState::Success;
}

Eigen::SparseMatrix<double>
JointLinearMPC_Smoother::blockDiagonalConcat_(const std::vector<std::reference_wrapper<const Eigen::SparseMatrix<double>>> &blocks)
{
    int totalRows = 0;
    int totalCols = 0;
    int nnz = 0;

    for (const auto &blockRef : blocks)
    {
        const auto &block = blockRef.get();
        totalRows += block.rows();
        totalCols += block.cols();
        nnz += block.nonZeros();
    }

    std::vector<Eigen::Triplet<double>> triplets;
    triplets.reserve(nnz);

    int rowOffset = 0;
    int colOffset = 0;
    for (const auto &blockRef : blocks)
    {
        const auto &block = blockRef.get();
        for (int k = 0; k < block.outerSize(); ++k)
            for (Eigen::SparseMatrix<double>::InnerIterator it(block, k); it; ++it)
                triplets.emplace_back(it.row() + rowOffset, it.col() + colOffset, it.value());
        rowOffset += block.rows();
        colOffset += block.cols();
    }

    Eigen::SparseMatrix<double> result(totalRows, totalCols);
    result.setFromTriplets(triplets.begin(), triplets.end());
    return result;
}

Eigen::SparseMatrix<double>
JointLinearMPC_Smoother::blockVerticalConcat_(const std::vector<std::reference_wrapper<const Eigen::SparseMatrix<double>>> &blocks)
{
    if (blocks.empty())
        return Eigen::SparseMatrix<double>();

    const int colCount = blocks[0].get().cols();
    int totalRows = 0;
    int totalNnz = 0;

    for (const auto &blockRef : blocks)
    {
        const auto &block = blockRef.get();
        if (block.cols() != colCount)
            throw std::invalid_argument("All matrices must have the same number of columns.");
        totalRows += block.rows();
        totalNnz += block.nonZeros();
    }

    std::vector<Eigen::Triplet<double>> triplets;
    triplets.reserve(totalNnz);

    int rowOffset = 0;
    for (const auto &blockRef : blocks)
    {
        const auto &block = blockRef.get();
        for (int k = 0; k < block.outerSize(); ++k)
            for (Eigen::SparseMatrix<double>::InnerIterator it(block, k); it; ++it)
                triplets.emplace_back(it.row() + rowOffset, it.col(), it.value());
        rowOffset += block.rows();
    }

    Eigen::SparseMatrix<double> result(totalRows, colCount);
    result.setFromTriplets(triplets.begin(), triplets.end());
    return result;
}

Eigen::VectorXd JointLinearMPC_Smoother::concatVectors_(const std::vector<std::reference_wrapper<const Eigen::VectorXd>> &vecs)
{
    size_t totalSize = 0;
    for (const auto &vRef : vecs)
        totalSize += vRef.get().size();
    Eigen::VectorXd result(totalSize);
    size_t pos = 0;
    for (const auto &vRef : vecs)
    {
        const auto &v = vRef.get();
        result.segment(pos, v.size()) = v;
        pos += v.size();
    }
    return result;
}

#endif /* USE_OSQP */
