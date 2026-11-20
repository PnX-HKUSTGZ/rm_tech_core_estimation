import os
from pathlib import Path
import time
import unittest

import cv2
from cv_bridge import CvBridge
from diagnostic_msgs.msg import DiagnosticArray
from geometry_msgs.msg import PoseStamped, TransformStamped
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, Shutdown, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare
import launch_testing
import launch_testing.actions
import launch_testing.asserts
import numpy as np
import rclpy
from rclpy.time import Time
from scipy.spatial.transform import Rotation
from sensor_msgs.msg import CameraInfo, Image
from tf2_msgs.msg import TFMessage
from tf2_ros.static_transform_broadcaster import StaticTransformBroadcaster
import yaml


RELEASE = Path(os.environ['TECH_CORE_GOLDEN_DIR'])
IMAGE_TOPIC = '/t6_camera/image_raw'
INFO_TOPIC = '/t6_calibration'


def generate_test_description():
    """运行已安装的 launch 文件，包括其相互独立的相机话题重映射。"""
    path = PathJoinSubstitution([FindPackageShare('tech_core_estimation_ros'), 'launch',
                                 'tech_core_estimation.launch.py'])
    estimator = IncludeLaunchDescription(PythonLaunchDescriptionSource(path), launch_arguments={
        'model_dir': str(RELEASE), 'image': IMAGE_TOPIC, 'camera_info': INFO_TOPIC,
    }.items())
    return LaunchDescription([
        estimator, launch_testing.actions.ReadyToTest(),
        TimerAction(period=60.0, actions=[Shutdown(reason='Camera replay deadline')]),
    ])


class TestCameraStream(unittest.TestCase):
    """回放同步输入，检查节点带时间戳的输出与失败计数。"""

    def setUp(self):
        rclpy.init()
        self.addCleanup(rclpy.shutdown)
        self.node = rclpy.create_node('t6_camera')
        self.addCleanup(self.node.destroy_node)
        self.bridge = CvBridge()
        self.image_pub = self.node.create_publisher(Image, IMAGE_TOPIC, 5)
        self.info_pub = self.node.create_publisher(CameraInfo, INFO_TOPIC, 5)
        self.poses = {}
        self.transforms = {}
        self.counts = {}
        self.node.create_subscription(PoseStamped, '/tech_core_estimation/core_pose', self.on_pose, 10)
        self.node.create_subscription(TFMessage, '/tf', self.on_tf, 10)
        self.node.create_subscription(DiagnosticArray, '/diagnostics', self.on_diagnostics, 10)

        static = TransformStamped()
        static.header.frame_id = 'base_link'
        static.child_frame_id = 'camera_optical'
        static.header.stamp = self.node.get_clock().now().to_msg()
        static.transform.translation.x = 1.25
        static.transform.translation.y = -0.5
        static.transform.rotation.z = static.transform.rotation.w = np.sqrt(0.5)
        self.static_tf = StaticTransformBroadcaster(self.node)
        self.static_tf.sendTransform(static)
        self.world_from_camera = np.array([[0, -1, 0, 1.25], [1, 0, 0, -0.5],
                                           [0, 0, 1, 0], [0, 0, 0, 1]])
        self.wait_for(lambda: self.image_pub.get_subscription_count() and
                      self.info_pub.get_subscription_count() and 'published' in self.counts,
                      'Estimator camera subscriptions and diagnostics', timeout=30)

    def on_pose(self, message):
        self.poses[Time.from_msg(message.header.stamp).nanoseconds] = message

    def on_tf(self, message):
        for transform in message.transforms:
            if transform.child_frame_id == 'tech_core':
                self.transforms[Time.from_msg(transform.header.stamp).nanoseconds] = transform

    def on_diagnostics(self, message):
        for status in message.status:
            if status.name.endswith('Pose estimation'):
                self.counts = {value.key: value.value for value in status.values}

    def wait_for(self, condition, description, timeout=5):
        deadline = time.monotonic() + timeout
        while not condition():
            self.assertLess(time.monotonic(), deadline, description)
            rclpy.spin_once(self.node, timeout_sec=0.05)

    def send(self, frame, case, optical_frame='camera_optical', stamp=None):
        if stamp is None:
            stamp = self.node.get_clock().now()
        image = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')
        image.header.stamp = stamp.to_msg()
        image.header.frame_id = optical_frame
        info = CameraInfo(header=image.header, width=image.width, height=image.height,
                          distortion_model=case.get('distortion_model', 'plumb_bob'), d=case['distortion'],
                          k=np.ravel(case['camera_matrix']).astype(float))
        self.info_pub.publish(info)
        self.image_pub.publish(image)
        return stamp.nanoseconds

    def expect_failure(self, frame, case, reasons, optical_frame='camera_optical', stamp=None):
        def failures():
            return sum(int(self.counts[reason]) for reason in reasons)

        previous = failures()
        stamp = self.send(frame, case, optical_frame, stamp)
        self.wait_for(lambda: failures() > previous, f'Diagnostic increment for {reasons}')
        self.assertNotIn(stamp, self.poses)
        self.assertNotIn(stamp, self.transforms)

    def expect_pose(self, frame, case):
        stamp = self.send(frame, case)
        self.wait_for(lambda: stamp in self.poses and stamp in self.transforms,
                      f'Stamped pose and TF for {case["image"]}')
        expected = self.world_from_camera @ np.array(case['camera_from_core'])
        message = self.poses[stamp]
        self.assertEqual(message.header.frame_id, 'base_link')
        position = message.pose.position
        translation = [position.x, position.y, position.z]
        q = message.pose.orientation
        rotation = Rotation.from_quat([q.x, q.y, q.z, q.w])
        self.assertLess(np.linalg.norm(translation - expected[:3, 3]), 1e-3)
        self.assertLess((rotation.inv() * Rotation.from_matrix(expected[:3, :3])).magnitude(), 1e-3)
        transform = self.transforms[stamp]
        self.assertEqual(transform.header, message.header)
        self.assertEqual(transform.transform.rotation, q)
        offset = transform.transform.translation
        self.assertEqual([offset.x, offset.y, offset.z], translation)

    def test_golden_stream_and_failures(self):
        golden = RELEASE / 'golden'
        cases = yaml.safe_load((golden / 'cases.yaml').read_text())
        for case in cases:
            with self.subTest(image=case['image']):
                frame = cv2.imread(str(golden / case['image']))
                self.assertIsNotNone(frame)
                if case['camera_from_core'] is not None:
                    self.expect_pose(frame, case)
                elif case['bbox_xyxy'] is None:
                    self.expect_failure(frame, case, ['no_detection'])
                elif sum(case['valid']) < 4:
                    self.expect_failure(frame, case, ['too_few_points'])
                else:
                    self.expect_failure(frame, case, ['no_consensus', 'degenerate', 'no_positive_depth'])

        case = next(case for case in cases if case['camera_from_core'] is not None)
        frame = cv2.imread(str(golden / case['image']))
        self.expect_failure(frame, case, ['invalid_stamp'], stamp=Time())
        self.expect_failure(frame, case, ['tf_timeout'], optical_frame='unconnected_camera_optical')
        self.expect_failure(np.zeros_like(frame), case, ['no_detection'])
        self.expect_failure(frame, {**case, 'camera_matrix': np.zeros((3, 3))}, ['invalid_calibration'])
        self.expect_pose(frame, case)
        self.expect_failure(frame, {**case, 'distortion': [0.0]}, ['invalid_calibration'])
        self.expect_failure(frame, {**case, 'distortion_model': 'equidistant'}, ['invalid_calibration'])
        self.expect_pose(frame, case)


@launch_testing.post_shutdown_test()
class TestShutdown(unittest.TestCase):
    """launch 关闭时独立组件正常退出。"""

    def test_exit_code(self, proc_info):
        launch_testing.asserts.assertExitCodes(proc_info)
