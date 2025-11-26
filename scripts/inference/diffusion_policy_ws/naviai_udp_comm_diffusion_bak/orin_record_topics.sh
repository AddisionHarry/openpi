#!/bin/bash

export ROS_IP=192.168.2.100
export ROS_MASTER_URI=http://192.168.217.1:11311
source /opt/ros/noetic/setup.bash
source /root/naviai_ws/devel/setup.bash

ROSBAG_LOG_PATH=/root/teleoperation_rosbag
mkdir -p $ROSBAG_LOG_PATH
if [ "$1" == "--record" ]; then
    rm -rf $ROSBAG_LOG_PATH/*.orig.*
    {
        MAX_FOLDER_SIZE=$((20 * 1024 * 1024 * 1024))
        current_size=$(du -sb "$ROSBAG_LOG_PATH" | awk '{print $1}')
        while true; do
            current_size=$(du -sb "$ROSBAG_LOG_PATH" | awk '{print $1}')
            if [ "$current_size" -gt "$MAX_FOLDER_SIZE" ]; then
                oldest_file=$(ls -t "$ROSBAG_LOG_PATH"/*.bag | tail -n 1)
                if [ -n "$oldest_file" ]; then
                    rm -f "$oldest_file"
                fi
            else
                break
            fi
        done
    } &
    rosbag record -o ${ROSBAG_LOG_PATH}/teleoperation --lz4 /joint_states /teleoperation_ctrl_cmd/final \
        /teleoperation_ctrl_cmd/recv_raw /hand_joint_states /calib_vel
elif [[ "$1" == "--clean" ]]; then
    {
        MAX_IDLE_TIME=60
        if [ ! -d "$ROSBAG_LOG_PATH" ] || [ -z "$(ls -A $ROSBAG_LOG_PATH)" ]; then
            echo "$ROSBAG_LOG_PATH is empty or does not exist. Exiting."
            exit 1
        fi
        latest_file=$(ls -t $ROSBAG_LOG_PATH | head -n 1)
        if [ -z "$latest_file" ]; then
            echo "No files found in $ROSBAG_LOG_PATH. Exiting."
            exit 1
        fi
        latest_file_path="$ROSBAG_LOG_PATH/$latest_file"
        last_modified=$(stat -c %Y "$latest_file_path")
        current_time=$(date +%s)
        idle_time=$((current_time - last_modified))
        if [ $idle_time -gt $MAX_IDLE_TIME ]; then
            echo "File $latest_file has not been modified in the last 60 seconds. Exiting."
            exit 0
        fi
        if [[ "$latest_file" == *.bag.active ]]; then
            mv "$latest_file_path" "${latest_file_path%.active}"
            echo "Renamed $latest_file to .bag and starting reindex."
            rosbag reindex "${latest_file_path%.active}"
        elif [[ "$latest_file" == *.bag ]]; then
            echo "Found .bag file, starting reindex."
            rosbag reindex "$latest_file_path"
        fi
        find $ROSBAG_LOG_PATH -name "*.orig.*" -exec rm -f {} \;
    } &
fi
