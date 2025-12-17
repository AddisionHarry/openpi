#!/usr/bin/env python3

import argparse
import rospy
from sensor_msgs.msg import JointState
from std_msgs.msg import Header

TEST_PUBLISH = False
TEST_PUBLISH_JOINT_POSITION = [0.0] * 4 + [0, -0.1, 0.6, 0.5, 0, 0, 0, 0] + [0, -0.1, -0.6, -0.5, 0, 0, 0, 0]

def publish_joint_states(robot_type: str):
    rospy.init_node('default_joint_state_publisher')
    if TEST_PUBLISH:
        pub = rospy.Publisher('/joint_states_feedback/raw', JointState, queue_size=10)
    else:
        pub = rospy.Publisher('/joint_states', JointState, queue_size=10)

    if robot_type == "H1_Pro":
        joint_names = [
            "Hip_Z_R", "Hip_X_R", "Hip_Y_R", "Knee_R", "Ankle_Y_R", "Ankle_X_R", "Hip_Z_L", "Hip_X_L", "Hip_Y_L",
            "Knee_L", "Ankle_Y_L", "Ankle_X_L", "A_Waist", "Shoulder_Y_R", "Shoulder_X_R", "Shoulder_Z_R", "Elbow_R",
            "Wrist_Z_R", "Wrist_Y_R", "Wrist_X_R", "Shoulder_Y_L", "Shoulder_X_L", "Shoulder_Z_L", "Elbow_L", "Wrist_Z_L",
            "Wrist_Y_L", "Wrist_X_L", "Neck_Z", "Neck_Y"
        ]
    elif robot_type == "WA1_0303":
        joint_names = [
            'Lifting_Z', 'Waist_Z', 'Waist_Y', 'Shoulder_Y_L', 'Shoulder_X_L', 'Shoulder_Z_L', 'Elbow_L', 'Wrist_Z_L',
            'Wrist_Y_L', 'Wrist_X_L', 'Shoulder_Y_R', 'Shoulder_X_R', 'Shoulder_Z_R', 'Elbow_R', 'Wrist_Z_R', 'Wrist_Y_R',
            'Wrist_X_R', 'Neck_Z', 'Neck_Y', 'Wheel_Y_R', 'Wheel_Y_L' , 'left_front_turn', 'left_front_scroll',
            'right_front_turn', 'right_front_scroll', 'left_back_turn', 'left_back_scroll', 'right_back_turn', 'right_back_scroll'
        ]
    elif robot_type == "WA2_A2_lite":
        joint_names = [
            'Pitch_Y_B', 'Pitch_Y_M', 'Waist_Z', 'Waist_Y',
            'Shoulder_Z_L', 'Shoulder_Y_L', 'Shoulder_X_L', 'Elbow_Z_L', 'Elbow_Y_L', 'Wrist_Z_L', 'Wrist_Y_L', 'Wrist_X_L',
            'Shoulder_Z_R', 'Shoulder_Y_R', 'Shoulder_X_R', 'Elbow_Z_R', 'Elbow_Y_R', 'Wrist_Z_R', 'Wrist_Y_R', 'Wrist_X_R',
            'Neck_Z', 'Neck_Y'
        ]
    else:
        rospy.logfatal(f"Get invalid robot type of {robot_type}, expected H1_Pro, WA1_0303, or WA2_A2_lite")
        exit(-10)

    rate = rospy.Rate(2)  # 2 Hz
    for i in range(5):
        joint_msg = JointState()
        joint_msg.header = Header()
        joint_msg.header.stamp = rospy.Time.now()
        joint_msg.name = joint_names
        if TEST_PUBLISH:
            if len(TEST_PUBLISH_JOINT_POSITION) < len(joint_names):
                joint_msg.position = TEST_PUBLISH_JOINT_POSITION + [0.0] * (len(joint_names) - len(TEST_PUBLISH_JOINT_POSITION))
            elif len(TEST_PUBLISH_JOINT_POSITION) >= len(joint_names):
                joint_msg.position = TEST_PUBLISH_JOINT_POSITION[:len(joint_names)]
        else:
            joint_msg.position = [0.0] * len(joint_names)
        joint_msg.velocity = [0.0] * len(joint_names)
        joint_msg.effort = [0.0] * len(joint_names)

        pub.publish(joint_msg)
        rospy.loginfo("Published joint state #%d", i + 1)
        rate.sleep()

if __name__ == '__main__':
    try:
        parser = argparse.ArgumentParser()
        parser.add_argument('--robot_type', type=str, required=True,
                            choices=["H1_Pro", "WA1_0303", "WA2_A2_lite"],
                            help='Specify robot type: H1_Pro, WA1_0303, or WA2_A2_lite')
        args = parser.parse_args()
        publish_joint_states(args.robot_type)
    except rospy.ROSInterruptException:
        pass
