%% viz_eval_h5.m
% Interactive visualization for evaluation H5 files
% Converted from viz_eval_h5.py
% Figures are shown directly (NOT saved)

%% ===================== User parameters =====================
h5_path = 'output.h5';     % <<< 修改为你的 h5 路径
out_dir = 'viz';           %#ok<NASGU> % 保留变量但不使用
top_k = 5;
chunk_stride = 10;
%% ===========================================================

assert(exist(h5_path, 'file') == 2, 'H5 file not found');

%% ---------------- Load episode-level summary ----------------
episode_idx = h5read(h5_path, '/episode_mse_summary/episode_idx');
mean_mse    = h5read(h5_path, '/episode_mse_summary/mean');

%% ---------------- Episode-level statistics ----------------
figure('Name','Episode Mean MSE Histogram');
histogram(mean_mse, 50);
xlabel('Mean Action MSE');
ylabel('Count');
title('Episode-level Mean Action MSE Distribution');
grid on;

[sorted_mse, order] = sort(mean_mse, 'descend');
figure('Name','Episodes Sorted by Mean MSE');
plot(sorted_mse, 'LineWidth', 1.5);
xlabel('Episode (sorted)');
ylabel('Mean Action MSE');
title('Episodes Sorted by Mean MSE');
grid on;

%% ---------------- Select worst episodes ----------------
top_k = min(top_k, numel(order));
worst_eps = episode_idx(order(1:top_k));

%% ---------------- Per-episode visualization ----------------
for e = 1:numel(worst_eps)
    ep = worst_eps(e);
    ep_name = sprintf('/episode_%06d', ep);

    % Load episode data
    step_mse = h5read(h5_path, [ep_name '/action_mse']);   % (T,)
    pred     = h5read(h5_path, [ep_name '/action']);       % (T,H,D)
    gt       = h5read(h5_path, [ep_name '/gt_action']);    % (T,D)

    joint_names = h5read(h5_path, [ep_name '/action_joint_names']);
    joint_names = string(joint_names');

    [T, H, D] = size(pred);
    dt = 1.0;

    % Round x-axis to nearest multiple of 50
    max_time = ceil(T / 50) * 50;
    T_cut = min(T, max_time);

    step_mse = step_mse(1:T_cut);
    pred     = pred(1:T_cut,:,:);
    gt       = gt(1:T_cut,:);

    %% ---------------- Plot ----------------
    figure('Name',sprintf('Episode %06d',ep), ...
           'Position',[100 100 1400 max(400, 300*D)]);

    colors = lines(max(D,10));

    for d = 1:D
        subplot(D,1,d); hold on;

        base_color = colors(d,:);

        % Predicted chunks
        for i = 1:chunk_stride:T_cut
            local_t = ((0:H-1) + (i-1)) * dt;
            y = squeeze(pred(i,:,d));

            for k = 1:H-1
                frac = ((k-1) / max(1,H-1))^0.8;
                lw = 1.5 - 1.2 * frac;
                fade_color = base_color * (1-frac) + frac;
                plot(local_t(k:k+1), y(k:k+1), ...
                     'Color', fade_color, 'LineWidth', lw);
            end
        end

        % Ground truth
        plot((0:T_cut-1)*dt, gt(:,d), 'k', 'LineWidth', 2);

        ylabel(joint_names(d), 'FontSize', 11);
        grid on;

        if d == 1
            title(sprintf('Episode %06d | stride=%d | GT + Pred + MSE', ...
                ep, chunk_stride), 'FontSize', 14);
        end

        % MSE (right axis)
        yyaxis right;
        area((0:T_cut-1)*dt, step_mse, ...
            'FaceColor','red', 'FaceAlpha',0.12, 'EdgeColor','none');

        mse_nz = step_mse(step_mse > 0);
        if ~isempty(mse_nz)
            ylim([0 max(prctile(mse_nz,95)*1.1, 0.01)]);
        else
            ylim([0 1]);
        end

        ylabel('Action MSE');
        set(gca,'YColor','red');
    end

    xlabel('Step');
    xlim([0 max_time]);

    % Ticks
    if max_time <= 100
        tick_spacing = 20;
    elseif max_time <= 200
        tick_spacing = 40;
    elseif max_time <= 500
        tick_spacing = 100;
    else
        tick_spacing = 200;
    end
    xticks(0:tick_spacing:max_time);

end

disp('[Done] All figures displayed.');
