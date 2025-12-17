#ifndef MPC_SOLVER_H
#define MPC_SOLVER_H

#if defined USE_OSQP

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <OsqpEigen/OsqpEigen.h>
#include <vector>

class LinearChainIntegrator
{
  public:
    static constexpr int stateDim = 4;
    static constexpr int inputDim = 1;

    double dt;

    explicit LinearChainIntegrator(double dt, bool inputInverse = true);
    virtual ~LinearChainIntegrator();
    LinearChainIntegrator(const Eigen::Matrix4d &A, const Eigen::Vector4d &B);
    LinearChainIntegrator(const LinearChainIntegrator &integrator, bool inputInverse);

    virtual Eigen::Vector4d step(const Eigen::Vector4d &x, double u) const;
    virtual const Eigen::Matrix4d &A(void) const;
    virtual const Eigen::Vector4d &B(void) const;

    void setInputInverse(void);

  private:
    Eigen::Matrix4d A_;
    Eigen::Vector4d B_;
};

class LinearMPC_Smoother
{
  public:
    using SparseMatrixXd = Eigen::SparseMatrix<double>;
    using Triplet = Eigen::Triplet<double>;

    enum class SolverState
    {
        Success,
        NotInited,
        UpdateBoundFailure,
        UpdateGradientFailure,
        SolverFailure,
        InitHessianMatrixFalure,
        InitLinearConstraintsMatrixFalure,
        InitSolverFalure,
        InnerGrammarFailure
    };

    static const char *toString(SolverState state);

    LinearMPC_Smoother(const LinearChainIntegrator &dynamics, int N, int Nu, const Eigen::MatrixXd &Q, const Eigen::MatrixXd &R,
                       double controlSmoothRate, const Eigen::MatrixXd &xmax, const Eigen::MatrixXd &xmin, const Eigen::MatrixXd &umax,
                       const Eigen::MatrixXd &umin);
    LinearMPC_Smoother(std::unique_ptr<LinearChainIntegrator> dynamics, int N, int Nu, const Eigen::MatrixXd &Q, const Eigen::MatrixXd &R,
                       double controlSmoothRate, const Eigen::MatrixXd &xmax, const Eigen::MatrixXd &xmin, const Eigen::MatrixXd &umax,
                       const Eigen::MatrixXd &umin);
    virtual ~LinearMPC_Smoother();
    SolverState initSolver(bool isVerbose = false);
    void setBounds(const Eigen::VectorXd &umin, const Eigen::VectorXd &umax, const Eigen::VectorXd &xmin, const Eigen::VectorXd &xmax);
    SolverState updateSolver(const Eigen::VectorXd &x0, const Eigen::VectorXd &r);
    virtual SolverState solve(const Eigen::VectorXd &x0, const Eigen::VectorXd &r, Eigen::VectorXd &uSeqOutput);
    Eigen::VectorXd limitOutput(const Eigen::VectorXd &uSeqOutput);
    virtual Eigen::VectorXd stepDynamics(const Eigen::VectorXd &x0, double u);
    virtual Eigen::VectorXd stepDynamics(const Eigen::VectorXd &x0, double u, double dt);

    Eigen::MatrixXd getHessianMatrix(void) const;
    const SparseMatrixXd &getHessianMatrixSparsed(void) const;
    Eigen::MatrixXd getPhiMatrix(void) const;
    const Eigen::VectorXd &getGradientVector(void) const;
    const Eigen::VectorXd &getBound(bool isLower) const;
    Eigen::MatrixXd getLinearConstraintMatrix(void) const;
    const SparseMatrixXd &getLinearConstraintMatrixSparsed(void) const;

    // Used for debug
    Eigen::VectorXd getUnconstrainedSolution(void) const;
    double getCurrentMPCCostValue(const Eigen::VectorXd &uSeqOutput) const;

  protected:
    std::unique_ptr<LinearChainIntegrator> dynamics_;
    std::unique_ptr<OsqpEigen::Solver> solver_;

  private:
    int N_, Nu_, nx_, nu_;
    double controlSmoothRate_;
    Eigen::MatrixXd Q_, R_;
    SparseMatrixXd tildeQ_, tildeR_;
    SparseMatrixXd P_, qLinear_;
    Eigen::VectorXd q_;
    SparseMatrixXd Tau_, Phi_;
    SparseMatrixXd A_;
    Eigen::VectorXd lbLinear_, ubLinear_;
    Eigen::VectorXd umin_, umax_, xmin_, xmax_;
    Eigen::VectorXd lowerBound_, upperBound_;
    bool inited_, initing_;

    Eigen::VectorXd cachedRef_;
    Eigen::Vector4d cachedR_;
    bool refValid_;

    void buildMatrices_(void);
    void buildCondensedMatrices_(void);
    void buildCostMatrices_(void);
    void buildConstraintMatrices_(void);
    Eigen::VectorXd buildReferenceVector_(const Eigen::VectorXd &r);

    bool updateOSQPGradient_(const Eigen::VectorXd &e);
    bool updateOSQPBounds_(const Eigen::VectorXd &x0, const Eigen::VectorXd &r);
};

// x = diag(xScale) * \tilde{x}, u = diag(uScale) * \tilde{u}
class NormalizedLinearChainIntegrator : public LinearChainIntegrator
{
  public:
    NormalizedLinearChainIntegrator(double dt, const Eigen::VectorXd &xscale, const Eigen::VectorXd &uscale, bool inputInverse = true);
    NormalizedLinearChainIntegrator(const NormalizedLinearChainIntegrator &integrator, bool inputInverse);
    ~NormalizedLinearChainIntegrator() override;

    Eigen::Vector4d stepOutter(const Eigen::Vector4d &x, double u) const;    // Unscaled system
    Eigen::Vector4d step(const Eigen::Vector4d &x, double u) const override; // Scaled system

    const Eigen::Matrix4d &A(void) const override; // Unscaled system
    const Eigen::Vector4d &B(void) const override; // Unscaled system

    const Eigen::Matrix4d &AOutter(void) const; // Unscaled system
    const Eigen::Vector4d &BOutter(void) const; // Unscaled system

    const Eigen::MatrixXd &getXScale(void) const;
    const Eigen::MatrixXd &getUScale(void) const;

  private:
    Eigen::MatrixXd xScale_, uScale_;
    Eigen::Matrix4d tildeA_;
    Eigen::Vector4d tildeB_;
};

class [[deprecated("After normalization the solver get unstable for numerical problems.")]] NormalizedLinearMPC_Smoother
    : public LinearMPC_Smoother
{
  public:
    // Q, R, ulim, xlim should pass unscaled parameters
    NormalizedLinearMPC_Smoother(const NormalizedLinearChainIntegrator &dynamics, int N, int Nu, const Eigen::MatrixXd &Q,
                                 const Eigen::MatrixXd &R, const Eigen::MatrixXd &xmax, const Eigen::MatrixXd &xmin,
                                 const Eigen::MatrixXd &umax, const Eigen::MatrixXd &umin);
    ~NormalizedLinearMPC_Smoother() override;
    SolverState solve(const Eigen::VectorXd &x0, const Eigen::VectorXd &r, Eigen::VectorXd &uSeqOutput) override;
    Eigen::VectorXd stepDynamics(const Eigen::VectorXd &x0, double u) override;

  private:
    Eigen::MatrixXd transformUSeqMat_;
};

class [[deprecated("Solver could not get correct results for unkown reason, but the calculation speed cost is not obversiouly "
                   "faster.")]] JointLinearMPC_Smoother
{
  public:
    using SolverState = LinearMPC_Smoother::SolverState;

    JointLinearMPC_Smoother(const LinearChainIntegrator &dynamics, int N, int Nu, const Eigen::MatrixXd &Q, const Eigen::MatrixXd &R,
                            const std::vector<Eigen::MatrixXd> &xmax, const std::vector<Eigen::MatrixXd> &xmin,
                            const std::vector<Eigen::MatrixXd> &umax, const std::vector<Eigen::MatrixXd> &umin);
    ~JointLinearMPC_Smoother() override;
    SolverState initSolver(bool isVerbose = false);
    SolverState solve(const std::vector<Eigen::VectorXd> &x0List, const std::vector<Eigen::VectorXd> &rList,
                      std::vector<Eigen::VectorXd> &uSeqOutputs);

  private:
    bool inited_;
    int N_, Nu_, numAgents_;
    std::vector<std::unique_ptr<LinearMPC_Smoother>> smoothers_;
    Eigen::SparseMatrix<double> bigP_, bigA_;
    Eigen::VectorXd bigq_, biglb_, bigub_;
    OsqpEigen::Solver solver_;
    std::vector<std::reference_wrapper<const Eigen::SparseMatrix<double>>> hessians_, constraints_;
    std::vector<std::reference_wrapper<const Eigen::VectorXd>> upperBounds_, lowerBounds_, gradients_;

    SolverState updateSolver_(const std::vector<Eigen::VectorXd> &x0List, const std::vector<Eigen::VectorXd> &rList);

    static Eigen::SparseMatrix<double> blockDiagonalConcat_(
        const std::vector<std::reference_wrapper<const Eigen::SparseMatrix<double>>> &blocks);
    static Eigen::SparseMatrix<double> blockVerticalConcat_(
        const std::vector<std::reference_wrapper<const Eigen::SparseMatrix<double>>> &blocks);
    static Eigen::VectorXd concatVectors_(const std::vector<std::reference_wrapper<const Eigen::VectorXd>> &vecs);
};

#endif /* USE_OSQP */

#endif /* MPC_SOLVER_H */
