close all; clear;

data_dir = "trajectory_data\20250929_231949\";
tcp_data = readtable(data_dir + "tcp_pose_data.csv");
target_pose_data = readtable(data_dir + "target_pose_data.csv");
control_cmd_data = readtable(data_dir + "control_command_data.csv");
network_action_data = readNetworkActionsMatrix(data_dir + "network_actions.h5");

plot_action = network_action_data;
plot_action.actions = network_action_data.actions(:, :, 1:3);
plot_pose_target = [target_pose_data{:, 1}, target_pose_data{:, 15:17}];
plot_tcp = [tcp_data{:, 1}, tcp_data{:, 9:11}];
figure; hold on; grid on;
xlabel('Time (s)'); ylabel('Action'); title('Target vs Controlled State vs Network Predicted Actions');
plot_network_vs_target(plot_action, plot_pose_target, plot_tcp);
hold off;
figure; hold on; grid on;
xlabel('Time (s)'); ylabel('Action'); title('Target vs Network Predicted Actions');
plot_network_vs_target(plot_action, plot_pose_target, []);
hold off;


function plot_network_vs_target(network_action_data, target_pose_data, real_pose_data)
    dt = 0.1;
    [nGroups, nActions, nDims] = size(network_action_data.actions);
    colors = lines(max(9, nDims));
    if real_pose_data
        t0 = min([min(network_action_data.timestamps), min(target_pose_data(:, 1)), min(real_pose_data(:, 1))]);
    else
        t0 = min([min(network_action_data.timestamps), min(target_pose_data(:, 1))]);
    end
    for d = 1:nDims
        baseColor = colors(d,:);
        for i = 1:nGroups
            local_t = (0:nActions-1) * dt + network_action_data.timestamps(i);
            y = squeeze(network_action_data.actions(i,:,d));  % 1¡ÁnActions
            for k = 1:(nActions-1)
                frac = ((k-1)/(nActions-1))^0.8;
                lw = 1.5 - 1.2 * frac;
                fadeColor = baseColor * (1 - frac) + frac;
                plot(local_t(k:k+1) - t0, y(k:k+1), '-', ...
                    'Color', fadeColor, 'LineWidth', lw, 'HandleVisibility', 'off');
            end
        end
    end
    for d = 1:nDims
        plot(target_pose_data(:, 1) - t0, target_pose_data(:, 1+d), ...
            'Color', colors(d+3, :) , 'LineWidth', 2);
        if real_pose_data
            plot(real_pose_data(:, 1) - t0, real_pose_data(:, 1+d), ...
                'Color', colors(d+6, :) , 'LineWidth', 1);
        end
    end
end


function data = readNetworkActionsMatrix(filename)
    info = h5info(filename);
    nGroups = length(info.Groups);
    leftActionsGroup = info.Groups(1).Groups(1);
    datasetNames = {leftActionsGroup.Datasets.Name};
    actionIdx = cellfun(@(s) sscanf(s, 'action_%d'), datasetNames);
    [~, sortIdx] = sort(actionIdx);
    datasetNames = datasetNames(sortIdx);
    sample = h5read(filename, [leftActionsGroup.Name '/' datasetNames{1}]);
    actionSize = length(sample);
    timestamps = zeros(nGroups,1);
    nActions = numel(datasetNames);
    actions = zeros(nGroups, nActions, actionSize);
    for i = 1:nGroups
        groupName = info.Groups(i).Name;
        ts = h5readatt(filename, groupName, 'timestamp');
        timestamps(i) = ts;
        leftActionsGroup = info.Groups(i).Groups(1);
        for j = 1:nActions
            path = [leftActionsGroup.Name '/' datasetNames{j}];
            vec = h5read(filename, path);
            vec = vec(:);
            actions(i,j,:) = vec;
        end
    end
    data.timestamps = timestamps;
    data.actions = actions;   % [nGroups ¡Á nActions ¡Á actionSize]
end

