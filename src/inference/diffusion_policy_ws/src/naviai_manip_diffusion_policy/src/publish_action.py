#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import time
from typing import List
from naviai_udp_comm.msg import TeleoperationUDPRaw

def set_target(neck_valid: bool, arm_valid: bool, hand_valid: bool, neck_target: List[float], arm_targets: List[List[float]],
               hand_targets: List[List[float]], waist_target: List[float]) -> TeleoperationUDPRaw:
    target = TeleoperationUDPRaw()
    target.recordingFlag = False
    target.controlFlag = (32 if neck_valid else 0) + (3 * 8 if arm_valid else 0) + (3 * 2 if hand_valid else 0)
    target.sendTick = target.recvTick = 0
    target.target.chassisCmd.linear.x = target.target.chassisCmd.linear.y = target.target.chassisCmd.linear.z = 0
    target.target.chassisCmd.angular.x = target.target.chassisCmd.angular.y = target.target.chassisCmd.angular.z = 0
    target.target.waistTargetForWheeled = waist_target[:2] + [0.0] * 2
    target.target.target.neckPosition = neck_target[:2]
    target.target.target.neckVelocity = [0.0] * 2
    target.target.target.leftArmPosition = arm_targets[0][:7] + [0.0]
    target.target.target.leftArmVelocity = [0.0] * 8
    target.target.target.rightArmPosition = arm_targets[1][:7] + [0.0]
    target.target.target.rightArmVelocity = [0.0] * 8
    target.target.target.leftHandPosition = hand_targets[0][:6]
    target.target.target.leftHandVelocity = [0.0] * 6
    target.target.target.rightHandPosition = hand_targets[1][:6]
    target.target.target.rightHandVelocity = [0.0] * 6
    target.target.target.waistPosition = [0.0] * 4
    target.target.target.waistVelocity = [0.0] * 4
    return target

if __name__ == "__main__":
    rospy.init_node("test_move_robot_node", anonymous=True)

    test_publisher = rospy.Publisher("/teleoperation_ctrl_cmd/recv_raw", TeleoperationUDPRaw, queue_size=10)

    while True:
        test_publisher.publish(set_target(True, True, True, [0.1, 0.4],
                                          [[0.0, 0.5, 0.0, -0.8, 0.0, 0.0, 0.0, 0.0], [-0.0, -0.5, 0.0, -0.5, 0.0, 0.0, 0.0, 0.0]],
                                          [[1.0, 1.0, 1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 0.0, 0.0, 0.0, 0.0]],
                                          [0.0, 0.5, 0.0, 0.0]))
        time.sleep(0.01)
