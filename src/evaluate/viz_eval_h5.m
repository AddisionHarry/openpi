%% viz_eval_h5_matlab_v2.m
% Interactive visualization for evaluation H5 files
% Can plot histograms, sorted curves, MSE curves, and optionally detailed joint curves

clear; clc;

%% -------------------- USER CONFIG --------------------
h5_path      = "evaluate.h5";
chunk_stride = 10;

% Optional: only plot these episode indices in detail (empty = skip detailed joint plots)
detailed_eps = [];  % e.g., [811, 812] or [] for no detailed plots
% -----------------------------------------------------

%% ========== Load episode-level summary ==========
episode_idx = h5read(h5_path, "/episode_mse_summary/episode_idx");
mean_mse    = h5read(h5_path, "/episode_mse_summary/mean");

%% ========== Histogram (popup) ==========
figure('Name','Episode Mean MSE Histogram');
histogram(mean_mse, 50);
xlabel('Mean Action MSE');
ylabel('Count');
title('Episode-level Mean Action MSE Distribution');
grid on;

%% ========== Sorted curve (popup) ==========
[~, order] = sort(mean_mse, 'descend');
figure('Name','Episodes Sorted by MSE');
plot(mean_mse(order), 'LineWidth', 1.5);
xlabel('Episode (sorted)');
ylabel('Mean Action MSE');
title('Episodes Sorted by Mean MSE');
grid on;

%% ========== MSE vs Episode Index (popup) ==========
figure('Name','Mean MSE vs Episode Index');
plot(episode_idx, mean_mse, 'b.-','LineWidth',1.5);
xlabel('Episode Index');
ylabel('Mean Action MSE');
title('Episode Mean MSE vs Episode Index');
grid on;

%% ========== Detailed visualization (optional) ==========
% Only plot if detailed_eps is specified
if ~isempty(detailed_eps)
    for e_idx = 1:numel(detailed_eps)

        ep = detailed_eps(e_idx);
        ep_name = sprintf("/episode_%06d", ep);
        fprintf("Processing %s\n", ep_name);

        % ---- Load data ----
        joint_names_raw = h5read(h5_path, ep_name + "/action_joint_names");
        joint_names = string(joint_names_raw');

        pred     = h5read(h5_path, ep_name + "/action");     % (D,H,T)
        gt       = h5read(h5_path, ep_name + "/gt_action");  % (D,T)
        step_mse = h5read(h5_path, ep_name + "/action_mse");

        [D,H,T] = size(pred);

        % ---- Time axis limit (rounded to 50) ----
        max_time = ceil(T / 50) * 50;
        T_cut = min(T, max_time);

        step_mse = step_mse(1:T_cut);
        pred     = pred(:,:,1:T_cut);
        gt       = gt(:,1:T_cut);

        time_full = 0:T_cut-1;

        % ---- Per joint: one figure per action dimension =====
        for d = 1:D
            figure('Name', sprintf('%s | %s', ep_name, joint_names(d)), ...
                   'Position',[100 100 1200 500]);
            hold on;

            % ---- Predicted chunks (fade) ----
            base_color = [0 0.4470 0.7410]; % MATLAB default blue
            for i = 1:chunk_stride:T_cut
                local_t = (0:H-1) + (i-1);
                y = squeeze(pred(d,:,i));

                for k = 1:H-1
                    frac = ((k-1) / max(1, H-1))^0.8;
                    lw   = 1.5 - 1.2 * frac;
                    fade_color = base_color*(1-frac) + frac*[1 1 1];
                    plot(local_t(k:k+1), y(k:k+1), 'Color', fade_color, 'LineWidth', lw);
                end
            end

            % ---- Predicted curve for legend only (invisible) ----
            p1 = plot(NaN,NaN,'Color',base_color,'LineWidth',1.5);

            % ---- Ground truth ----
            p2 = plot(time_full, gt(d,:), 'k', 'LineWidth', 2);

            % ---- MSE (right axis) ----
            yyaxis right
            p3 = area(time_full, step_mse, 'FaceColor','red', ...
                      'FaceAlpha',0.12, 'EdgeColor','none');
            ylabel('Action MSE','Interpreter','none');
            ylim([0 max(prctile(step_mse(step_mse>0),95)*1.1, 0.01)]);
            ax = gca;
            ax.YColor = 'red';

            % ---- Left axis config ----
            yyaxis left
            ylabel(joint_names(d),'Interpreter','none');
            xlabel('Step');
            grid on;

            title(sprintf('%s | Joint: %s | stride=%d', ...
                  ep_name, joint_names(d), chunk_stride), ...
                  'Interpreter','none');

            % ---- Legend with correct colors ----
            legend([p1,p2,p3], {'Predicted Chunks','Ground Truth','Action MSE'}, ...
                   'Location','northwest');

            xlim([0 max_time]);
            hold off;
        end

    end
end

fprintf("[Done] Interactive visualization finished.\n");
