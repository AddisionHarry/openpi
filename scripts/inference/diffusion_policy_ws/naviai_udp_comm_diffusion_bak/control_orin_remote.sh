#!/bin/bash

SCRIPT_DIR=$(realpath $(dirname "$BASH_SOURCE"))
source "$SCRIPT_DIR/../../environments.conf"
TARGET_IP=$ORIN_IP
TARGET_USER=$REMOTE_USER
TARGET_PASS=$REMOTE_PASSWD

$SCRIPT_DIR/update_configs.sh

if [ "$RUN_LOCAL" = "true" ]; then
    echo "Running locally, skipping target IP check."
else
    if ! ping -c 2 -W 1 "$TARGET_IP" >/dev/null 2>&1; then
        echo -e "\e[31mError: Unable to reach target IP ($TARGET_IP). Please check network connection.\e[0m" >&2
        exit 1
    fi
fi

remote_cmd() {
    sshpass -p "$TARGET_PASS" ssh -t $TARGET_USER@$TARGET_IP "$@"
}

remote_rsync() {
    sshpass -p "$TARGET_PASS" rsync -av $1 $TARGET_USER@$TARGET_IP:$2
}

if [[ "$1" == "--start-comm" ]]; then
    echo "Starting communication..."
    remote_rsync "$SCRIPT_DIR" "/home/$TARGET_USER/workspace/teleoperation/"
    remote_rsync "$SCRIPT_DIR/../system_state_msgs" "/home/$TARGET_USER/teleoperation/"

    if [[ "$2" == "--check-drivers" ]]; then
        remote_cmd 'docker exec -it naviai_manip bash -c ./start_roscore_and_remote_bash.sh'
    fi

    remote_cmd 'docker exec -it naviai_manip bash -c ./teleoperation/naviai_udp_comm/orin_udp_startup.sh'

elif [[ "$1" == "--kill-node" ]]; then
    echo "Killing communication node..."
    remote_cmd 'docker exec -it naviai_manip bash -c "\
        source /opt/ros/noetic/setup.bash && \
        rosnode kill /teleoperation_communication_control_node /teleoperation_communication_image_feedback_node"'

    if [[ "$2" == "--kill-drivers" ]]; then
        remote_cmd 'docker exec -it naviai_manip bash -c ./kill_roscore_and_remote_bash.sh'
    fi

elif [[ "$1" == "--sync-pack" ]]; then
    echo "Synchronizing packages..."
    remote_rsync "$SCRIPT_DIR" "/home/$TARGET_USER/teleoperation/"
    remote_rsync "$SCRIPT_DIR/../system_state_msgs" "/home/$TARGET_USER/teleoperation/"
    remote_rsync "$SCRIPT_DIR/../naviai_manip_types" "/home/$TARGET_USER/teleoperation/"

else
    echo "Usage: $0 {--start-comm|--kill-node|--sync-pack}"
    exit 1
fi
