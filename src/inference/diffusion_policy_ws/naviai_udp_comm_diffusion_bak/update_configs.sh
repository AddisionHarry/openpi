#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source ${SCRIPT_DIR}/../../environments.conf

$SCRIPT_DIR/modify_target_ip.sh $CURRENT_IP $ORIN_IP

cp -f $SCRIPT_DIR/orin_udp_startup.sh.example $SCRIPT_DIR/orin_udp_startup.sh
sed -i \
    -e "s/export ROS_IP=192.168.2.100/export ROS_IP=${ORIN_ROS_IP}/g" \
    -e "s/target_ip:=192.168.88.100/target_ip:=${CURRENT_IP}/g" \
    -e "s/control_wheeled_robot_move:=false/control_wheeled_robot_move:=${WHEELED_CONTROL_MOVE_FROM_VR}/g" \
    -e "s/use_robot_type:=H1_Pro/use_robot_type:=${USE_ROBOT_TYPE}/g" \
    -e "s/step_frequency:=300/step_frequency:=${CONTROL_STEP_FREQUENCY}/g" \
    -e "s/command_frequency:=300/command_frequency:=${STEP_FREQUENCY}/g" \
    -e "s/control_wheeled_robot_move:=false/control_wheeled_robot_move:=${WHEELED_CONTROL_MOVE_FROM_VR}/g" \
"$SCRIPT_DIR/orin_udp_startup.sh"

cp -f $SCRIPT_DIR/orin_record_topics.sh.example $SCRIPT_DIR/orin_record_topics.sh
sed -i \
    -e "s/export ROS_IP=192.168.2.100/export ROS_IP=${ORIN_ROS_IP}/g" \
    -e "s/export ROS_MASTER_URI=http:\/\/192.168.217.1:11311/export ROS_MASTER_URI=http:\/\/${ROBOT_ROS_MASTER_URI}:11311/g" \
"$SCRIPT_DIR/orin_udp_startup.sh"

cp -f $SCRIPT_DIR/start_rtmp_streamer.sh.example $SCRIPT_DIR/start_rtmp_streamer.sh
sed -i -E "s/--ip [0-9]+\.[0-9]+\.[0-9]+\.[0-9]+/--ip $CURRENT_IP/g" "$SCRIPT_DIR/start_rtmp_streamer.sh"
if [ "$RUN_LOCAL" == "true" ]; then
    sed -i 's/--orin//g' "$SCRIPT_DIR/start_rtmp_streamer.sh"
fi
