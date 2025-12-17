#!/usr/bin/env python3

import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib
import rospy
import cv2
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
import argparse
import threading

CAMERA_CONFIGS = [
    {
        "topic": "/realsense_up/color/image_raw",
        "size": (1280, 720),
        "stream_name": "chest_cam"
    },
    {
        "topic": "/img/CAM_A/image_raw",
        "size": (1152, 720),
        "stream_name": "head_cam_a"
    },
    {
        "topic": "/img/CAM_B/image_raw",
        "size": (1152, 720),
        "stream_name": "head_cam_b"
    },
    {
        "topic": "/img/left_wrist/image_raw",
        "size": (640, 480),
        "stream_name": "left_wrist_cam"
    },
    {
        "topic": "/img/right_wrist/image_raw",
        "size": (640, 480),
        "stream_name": "right_wrist_cam"
    }
]

class CameraStream:
    def __init__(self, config, rtmp_ip: str, use_orin: bool):
        self.config = config
        self.last_recv_image_tick = rospy.Time.now()
        self.bridge = CvBridge()
        self.use_orin = use_orin
        self.rtmp_location = f"rtmp://{rtmp_ip}:1935/live/{config['stream_name']}"
        self._init_pipeline()
        rospy.Subscriber(config["topic"], Image, self.image_callback)
        rospy.loginfo(f"Start receiving images from topic {config['topic']} and streaming to {self.rtmp_location}")

    def _init_pipeline(self):
        if self.use_orin:
            self.pipeline = Gst.parse_launch(f"""
                appsrc name=ros_cam caps="video/x-raw,format=BGR,width={self.config['size'][0]},height={self.config['size'][1]},framerate=30/1" !
                videoconvert ! video/x-raw,format=I420 !
                nvvidconv ! video/x-raw(memory:NVMM),format=I420 !
                nvv4l2h264enc bitrate=4000000 preset-level=4 insert-sps-pps=1 control-rate=1 !
                h264parse config-interval=1 ! flvmux streamable=true ! rtmpsink location={self.rtmp_location} sync=false
            """)
        else:
            self.pipeline = Gst.parse_launch(f"""
                appsrc name=ros_cam caps="video/x-raw,format=BGR,width={self.config['size'][0]},height={self.config['size'][1]},framerate=30/1" !
                videoconvert ! video/x-raw,format=I420 !
                x264enc bitrate=4000 speed-preset=ultrafast tune=zerolatency !
                h264parse config-interval=1 ! flvmux streamable=true !
                rtmpsink location={self.rtmp_location} sync=false
            """)
        self.appsrc = self.pipeline.get_by_name("ros_cam")
        self.pipeline.set_state(Gst.State.PLAYING)

    def image_callback(self, msg):
        try:
            self.last_recv_image_tick = rospy.Time.now()
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            gst_buffer = Gst.Buffer.new_wrapped(cv_image.tobytes())
            self.appsrc.emit("push-buffer", gst_buffer)
        except Exception as e:
            rospy.logerr(f"Get error while processing image from {self.config['topic']}: {str(e)}")

    def check_image_receiving(self) -> bool:
        return (rospy.Time.now() - self.last_recv_image_tick).to_sec() <= 1

    def cleanup(self):
        self.pipeline.send_event(Gst.Event.new_eos())
        self.pipeline.set_state(Gst.State.NULL)

class GstRTMPPublisher:
    def __init__(self, rtmp_ip: str, use_orin: bool):
        rospy.init_node('gst_rtmp_publisher')
        Gst.init(None)
        self.streams = [CameraStream(config, rtmp_ip, use_orin) for config in CAMERA_CONFIGS]
        rospy.on_shutdown(self.cleanup)
        threading.Thread(target=self.__monitor_streams, daemon=True).start()

    def __monitor_streams(self) -> None:
        while not rospy.is_shutdown():
            rospy.Rate(10).sleep()
            for stream in self.streams:
                if not stream.check_image_receiving():
                    rospy.logerr(f"No image received from {stream.config['topic']} for {(rospy.Time.now() - stream.last_recv_image_tick).to_sec()} seconds")

    def cleanup(self):
        rospy.loginfo("Stopping all GStreamer pipelines")
        for stream in self.streams:
            stream.cleanup()
        if hasattr(self, 'loop'):
            self.loop.quit()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GStreamer RTMP Publisher with ROS Image')
    parser.add_argument('--ip', type=str, default='192.168.88.100', help='RTMP server IP address')
    parser.add_argument('--orin', action='store_true', help='Use Jetson encode hardware interface')
    args = parser.parse_args()

    try:
        publisher = GstRTMPPublisher(args.ip, args.orin)
        loop = GLib.MainLoop()
        publisher.loop = loop  # Store reference for cleanup
        loop.run()
    except rospy.ROSInterruptException:
        pass


'''
Test local
export CANDIDATE=127.0.0.1
PROJECT_PATH="/home/leehe/Work/TeleVision-ThreeJS"
docker run --rm -it -p 1935:1935 -p 1985:1985 -p 8080:8080 -p 1990:1990 -p 8088:8088 \
    --env CANDIDATE=$CANDIDATE -p 8000:8000/udp \
    -v ${PROJECT_PATH}/ros/naviai_udp_comm/config/https.docker.conf:/usr/local/srs/conf/https.docker.conf \
    -v ${PROJECT_PATH}/asserts/SSL/server.crt:/usr/local/srs/server.crt \
    -v ${PROJECT_PATH}/asserts/SSL/server.key:/usr/local/srs/server.key \
    registry.cn-hangzhou.aliyuncs.com/ossrs/srs:5 ./objs/srs -c conf/https.docker.conf

export CANDIDATE=192.168.88.100
PROJECT_PATH="/home/teleoperation/TeleVision-ThreeJS"
docker run --rm -it -p 1935:1935 -p 1985:1985 -p 8080:8080 -p 1990:1990 -p 8088:8088 \
    --env CANDIDATE=$CANDIDATE -p 8000:8000/udp \
    -v ${PROJECT_PATH}/ros/naviai_udp_comm/config/https.docker.conf:/usr/local/srs/conf/https.docker.conf \
    -v ${PROJECT_PATH}/asserts/SSL/server.crt:/usr/local/srs/server.crt \
    -v ${PROJECT_PATH}/asserts/SSL/server.key:/usr/local/srs/server.key \
    registry.cn-hangzhou.aliyuncs.com/ossrs/srs:5 ./objs/srs -c conf/https.docker.conf

gst-launch-1.0 v4l2src device=/dev/video4 ! video/x-raw,format=YUY2,width=1280,height=720,framerate=30/1 \
    ! videoconvert ! x264enc bitrate=4000 speed-preset=ultrafast tune=zerolatency ! h264parse ! flvmux \
    ! rtmpsink location="rtmp://172.16.8.36:1935/live/chest_camera"
'''
