clear; close all;
%% Parameter settings
N = 1500;                % Total time steps of simulation
Ts = 1 / 50;             % Step frequency
x0 = [-2; -2; 0.3; -1];  % Initial state of MPC
predict_horizon = 150;   % Predict horizon for MPC
control_horizon = 15;    % Control horizon for MPC

% Reference signal
ref_func = @(t) ...
    ((t < 8.5) .* [sin(0.7*t)+1; 0.7*cos(0.7*t); 0; 0]) + ...
    (((t >= 8.5)&(t < 12.5)) .* [3; 0; 0; 0]) + ...
    (((t >= 12.5)&(t < 17)) .* [-3 + (t-12.5) * 1.05; 1.05; 0; 0]) + ...
    (((t >= 17)&(t < 21.5)) .* [-3; 0; 0; 0]) + ...
    ((t >= 21.5) .* [-cos(0.65*(t-22))-1; 0.65 * sin(0.65*(t-22)); 0; 0]);
%% MPC Controller
x = x0;
x_log = zeros(4, N);
u_log = zeros(1, N);
r_log = zeros(4, N);

tic;
for k = 1:N
    t = (k-1)*Ts;
    r = ref_func(t);
    u = mpc_controller(x, r, Ts, predict_horizon, control_horizon);
    x = simulate_continuous_step_persistent(x, u, Ts);
    x_mpc.Plant = x;
    x_log(:,k) = x;
    u_log(k) = u;
    r_log(:,k) = r;
end
total_time = toc;
time = (0:N-1)*Ts;
figure;
subplot(5,1,1)
plot(time, x_log(1,:), 'b', time, r_log(1,:), 'r--')
ylabel('Position'); legend('x_1','r_1')
subplot(5,1,2)
plot(time, x_log(2,:), 'b', time, r_log(2,:), 'r--')
ylabel('Velocity'); legend('x_2','r_2')
subplot(5,1,3)
plot(time, x_log(3,:), 'b', time, r_log(3,:), 'r--')
ylabel('Acceleration'); legend('x_3','r_3')
subplot(5,1,4)
plot(time, x_log(4,:), 'b', time, r_log(4,:), 'r--')
ylabel('Jerk'); legend('x_4','r_4')
subplot(5,1,5)
plot(time, u_log, 'k')
ylabel('Control Input u'); xlabel('Time (s)'); legend('u')
sgtitle("MPC Simulation without tracking")
fprintf("MPC Average calculate time: %.6f milliseconds per iteration\n", total_time / N * 1000);

function u = mpc_controller(x, r, Ts, predict_horizon, control_horizon)
persistent mpcobj x_mpc
if isempty(mpcobj)
    A = [0, 1, 0, 0;
        0, 0, 1, 0;
        0, 0, 0, 1;
        0, 0, 0, 0];
    B = [0; 0; 0; 1];
    C = eye(4);
    D = zeros(4, 1);
    sys_d = c2d(ss(A, B, C, D), Ts);
    mpcobj = mpc(sys_d, Ts, predict_horizon, control_horizon);
    mpcobj.Weights.OutputVariables = [25 0.5 0 0];
    mpcobj.Weights.ManipulatedVariables = 5e-2;
    mpcobj.Weights.ManipulatedVariablesRate = 0;
    mpcobj.MV.Min = -200;
    mpcobj.MV.Max = 200;
    mpcobj.OV(1).Min = -3;   mpcobj.OV(1).Max = 3;
    mpcobj.OV(2).Min = -2.8; mpcobj.OV(2).Max = 2.8;
    mpcobj.OV(3).Min = -4.6; mpcobj.OV(3).Max = 4.6;
    mpcobj.OV(4).Min = -50; mpcobj.OV(4).Max = 50;
    x_mpc = mpcstate(mpcobj);
    tic; % Remove the initilization of mpc time
end
x_mpc.Plant = x;
u = mpcmove(mpcobj, x_mpc, x, r);
end

function x_next = simulate_continuous_step_persistent(x, u, t)
persistent A B
if isempty(A) || isempty(B)
    A = [0, 1, 0, 0;
        0, 0, 1, 0;
        0, 0, 0, 1;
        0, 0, 0, 0];
    B = [0; 0; 0; 1];
end
M = expm([A, B; zeros(size(B,2), size(A,1)+size(B,2))] * t);
Phi = M(1:size(A,1), 1:size(A,2));
Gamma = M(1:size(A,1), size(A,2)+1:end);
x_next = Phi * x + Gamma * u;
end

%% Tracking MPC Controller
x = x0;
x_log = zeros(4, N);
u_log = zeros(1, N);
r_log = zeros(4, N);
tic;
for k = 1:N
    t = (k-1)*Ts;
    r = ref_func(t);
    u = mpc_controller_tracking(r - x, Ts, predict_horizon, control_horizon);
    x = simulate_continuous_step_persistent(x, u, Ts);
    x_mpc.Plant = x;
    x_log(:,k) = x;
    u_log(k) = u;
    r_log(:,k) = r;
end
total_time = toc;
time = (0:N-1)*Ts;
figure;
subplot(5,1,1)
plot(time, x_log(1,:), 'b', time, r_log(1,:), 'r--')
ylabel('Position'); legend('x_1','r_1')
subplot(5,1,2)
plot(time, x_log(2,:), 'b', time, r_log(2,:), 'r--')
ylabel('Velocity'); legend('x_2','r_2')
subplot(5,1,3)
plot(time, x_log(3,:), 'b', time, r_log(3,:), 'r--')
ylabel('Acceleration'); legend('x_3','r_3')
subplot(5,1,4)
plot(time, x_log(4,:), 'b', time, r_log(4,:), 'r--')
ylabel('Jerk'); legend('x_4','r_4')
subplot(5,1,5)
plot(time, u_log, 'k')
ylabel('Control Input u'); xlabel('Time (s)'); legend('u')
sgtitle("MPC Simulation with tracking")
fprintf("MPC Average calculate time: %.6f milliseconds per iteration\n", total_time / N * 1000);

figure;
subplot(4,1,1)
plot(time, x_log(1,:), 'b', time, r_log(1,:), 'r--')
ylabel('Position'); legend('x_1','r_1')
subplot(4,1,2)
plot(time, x_log(2,:), 'b', time, r_log(2,:), 'r--')
ylabel('Velocity'); legend('x_2','r_2')
subplot(4,1,3)
plot(time, x_log(3,:), 'b', time, r_log(3,:), 'r--')
ylabel('Acceleration'); legend('x_3','r_3')
subplot(4,1,4)
plot(time, x_log(4,:), 'b', time, r_log(4,:), 'r--')
ylabel('Jerk'); legend('x_4','r_4')
sgtitle("MPC Simulation with tracking")

function u = mpc_controller_tracking(x, Ts, predict_horizon, control_horizon)
persistent mpcobj x_mpc
if isempty(mpcobj)
    A = [0, 1, 0, 0;
        0, 0, 1, 0;
        0, 0, 0, 1;
        0, 0, 0, 0];
    B = [0; 0; 0; -1];
    C = eye(4);
    D = zeros(4, 1);
    sys_d = c2d(ss(A, B, C, D), Ts);
    mpcobj = mpc(sys_d, Ts, predict_horizon, control_horizon);
    mpcobj.Weights.OutputVariables = [25 0.5 0 0];
    mpcobj.Weights.ManipulatedVariables = 5e-2;
    mpcobj.Weights.ManipulatedVariablesRate = 0;
    mpcobj.MV.Min = -200;
    mpcobj.MV.Max = 200;
    mpcobj.OV(1).Min = -3;   mpcobj.OV(1).Max = 3;
    mpcobj.OV(2).Min = -2.8; mpcobj.OV(2).Max = 2.8;
    mpcobj.OV(3).Min = -4.6; mpcobj.OV(3).Max = 4.6;
    mpcobj.OV(4).Min = -50; mpcobj.OV(4).Max = 50;
    x_mpc = mpcstate(mpcobj);
    tic; % Remove the initilization of mpc time
end
x_mpc.Plant = x;
u = mpcmove(mpcobj, x_mpc, x, zeros(size(x)));
end

%% Display C++ MPC Result
cpp_mpc_res = readtable("~/Work/TeleVision-ThreeJS/logs/test_mpc.txt");
time = cpp_mpc_res.t;
figure;
subplot(5,1,1)
plot(time, cpp_mpc_res.p, 'b', time, cpp_mpc_res.rp, 'r--')
ylabel('Position'); legend('x_1','r_1')
subplot(5,1,2)
plot(time, cpp_mpc_res.v, 'b', time, cpp_mpc_res.rv, 'r--')
ylabel('Velocity'); legend('x_2','r_2')
subplot(5,1,3)
plot(time, cpp_mpc_res.a, 'b', time, cpp_mpc_res.ra, 'r--')
ylabel('Acceleration'); legend('x_3','r_3')
subplot(5,1,4)
plot(time, cpp_mpc_res.j, 'b', time, cpp_mpc_res.rj, 'r--')
ylabel('Jerk'); legend('x_4','r_4')
subplot(5,1,5)
plot(time, cpp_mpc_res.s, 'k')
ylabel('Control Input u'); xlabel('Time (s)'); legend('u')
sgtitle("C++ MPC Simulation with tracking")

figure;plot(1:length(cpp_mpc_res.t), cpp_mpc_res.solveTime_us, 'b');
yline(mean(cpp_mpc_res.solveTime_us), 'r--', 'LineWidth', 1.5);
ylabel("Solve Time/us"); xlabel("Iteration");
title("C++ OSQP-Eigen Solver Solve Time")
