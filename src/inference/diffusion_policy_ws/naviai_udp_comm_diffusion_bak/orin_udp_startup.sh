#!/bin/bash

cleanup() {
    pkill -9 naviai_ud
}

trap cleanup INT TERM QUIT

export ROS_IP=192.168.217.100
export ROS_MASTER_URI=http://192.168.217.1:11311
echo $ROS_IP
echo $ROS_MASTER_URI
cd /root/naviai_ws
source /opt/ros/noetic/setup.bash

NO_BUILD=false
for arg in "$@"
do
    if [[ "$arg" == "--no-build" ]]; then
        NO_BUILD=true
    fi
done

if [ "$NO_BUILD" = false ]; then
    # catkin build -j6
    # if catkin build -j6; then
    catkin_make -j6
    if catkin_make -j6; then
        echo -e "Compile success!"
    else
        echo -e "\e[31mCompile failed!\e[0m"
        exit 1
    fi
fi

source devel/setup.bash
cleanup
roslaunch naviai_udp_comm orin_teleoperation.launch use_robot_type:=WA1_0303 step_frequency:=300 \
    command_frequency:=180 control_wheeled_robot_move:=true target_ip:=192.168.217.10
