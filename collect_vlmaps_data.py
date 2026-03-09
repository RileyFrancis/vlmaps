import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from sensor_msgs.msg import Image, PointCloud2
from nav_msgs.msg import Odometry
from cv_bridge import CvBridge
import cv2, numpy as np, os, message_filters
import sensor_msgs_py.point_cloud2 as pc2
import tf2_ros
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener

SAVE_DIR = '/ros2_ws/src/vlmaps/data/go2_scene_1'
RGB_TOPIC   = '/camera/image_raw'
CLOUD_TOPIC = '/point_cloud2'
ODOM_TOPIC  = '/odom'

FX, FY = 864.39938, 863.73849
CX, CY = 639.19798, 373.28118
IMG_W, IMG_H = 1280, 720

BEST_EFFORT_QOS = QoSProfile(
    reliability=ReliabilityPolicy.BEST_EFFORT,
    history=HistoryPolicy.KEEP_LAST,
    depth=10,
    durability=DurabilityPolicy.VOLATILE
)

class Collector(Node):
    def __init__(self):
        super().__init__('vlmaps_collector')
        self.bridge = CvBridge()
        self.rgb_dir   = os.path.join(SAVE_DIR, 'rgb');   os.makedirs(self.rgb_dir, exist_ok=True)
        self.depth_dir = os.path.join(SAVE_DIR, 'depth'); os.makedirs(self.depth_dir, exist_ok=True)
        self.poses_f   = open(os.path.join(SAVE_DIR, 'poses.txt'), 'w')
        self.idx = 0

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        rgb_sub   = message_filters.Subscriber(self, Image,        RGB_TOPIC,   qos_profile=BEST_EFFORT_QOS)
        cloud_sub = message_filters.Subscriber(self, PointCloud2,  CLOUD_TOPIC, qos_profile=BEST_EFFORT_QOS)
        odom_sub  = message_filters.Subscriber(self, Odometry,     ODOM_TOPIC,  qos_profile=BEST_EFFORT_QOS)

        self.ts = message_filters.ApproximateTimeSynchronizer(
            [rgb_sub, cloud_sub, odom_sub], queue_size=20, slop=0.5)
        self.ts.registerCallback(self.cb)
        self.get_logger().info(f'Saving to {SAVE_DIR} ...')
        self.get_logger().info('Waiting for messages on all 3 topics...')

    def get_lidar_to_cam(self):
        try:
            t = self.tf_buffer.lookup_transform(
                'front_camera',
                'base_link',
                rclpy.time.Time()
            )
            tx, ty, tz = t.transform.translation.x, t.transform.translation.y, t.transform.translation.z
            x, y, z, w = t.transform.rotation.x, t.transform.rotation.y, t.transform.rotation.z, t.transform.rotation.w
            R = np.array([
                [1-2*(y*y+z*z),   2*(x*y-z*w),   2*(x*z+y*w)],
                [  2*(x*y+z*w), 1-2*(x*x+z*z),   2*(y*z-x*w)],
                [  2*(x*z-y*w),   2*(y*z+x*w), 1-2*(x*x+y*y)]
            ])
            T = np.eye(4)
            T[:3, :3] = R
            T[:3,  3] = [tx, ty, tz]
            return T
        except Exception as e:
            self.get_logger().warn(f'TF lookup failed: {e}')
            return None

    def project_cloud_to_depth(self, cloud_msg):
        depth_img = np.zeros((IMG_H, IMG_W), dtype=np.float32)
        T = self.get_lidar_to_cam()
        if T is None:
            return depth_img

        pts = list(pc2.read_points(cloud_msg, field_names=('x','y','z'), skip_nans=True))
        if len(pts) == 0:
            return depth_img

        points = np.array([[p[0], p[1], p[2], 1.0] for p in pts])
        pts_cam = (T @ points.T).T[:, :3]

        mask = pts_cam[:, 2] > 0.1
        pts_cam = pts_cam[mask]
        if len(pts_cam) == 0:
            return depth_img

        u = (FX * pts_cam[:, 0] / pts_cam[:, 2] + CX).astype(int)
        v = (FY * pts_cam[:, 1] / pts_cam[:, 2] + CY).astype(int)
        z = pts_cam[:, 2]

        valid = (u >= 0) & (u < IMG_W) & (v >= 0) & (v < IMG_H)
        depth_img[v[valid], u[valid]] = z[valid]
        return depth_img

    def cb(self, rgb_msg, cloud_msg, odom_msg):
        # Save RGB
        rgb = self.bridge.imgmsg_to_cv2(rgb_msg, 'bgr8')
        cv2.imwrite(f'{self.rgb_dir}/{self.idx:06d}.png', rgb)

        # Save depth
        depth = self.project_cloud_to_depth(cloud_msg)
        np.save(f'{self.depth_dir}/{self.idx:06d}.npy', depth)

        # Save pose
        p = odom_msg.pose.pose.position
        q = odom_msg.pose.pose.orientation
        self.poses_f.write(f'{p.x} {p.y} {p.z} {q.x} {q.y} {q.z} {q.w}\n')
        self.poses_f.flush()

        self.idx += 1
        self.get_logger().info(f'Frame {self.idx} saved | depth nonzero: {np.count_nonzero(depth)} px')

def main():
    rclpy.init()
    node = Collector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.poses_f.close()
    node.get_logger().info(f'Done. {node.idx} frames saved to {SAVE_DIR}')

if __name__ == '__main__':
    main()