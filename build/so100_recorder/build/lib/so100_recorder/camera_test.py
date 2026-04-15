import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

class CameraViewerNode(Node):
    def __init__(self):
        super().__init__('camera_viewer')
        
        self.bridge = CvBridge()
        
        # Subscribe to both cameras using the exact topics from your last screenshot
        self.sub_overhead = self.create_subscription(Image, '/camera_overhead', self.overhead_callback, 10)
        self.sub_gripper = self.create_subscription(Image, '/gripper_Cameras', self.gripper_callback, 10)
        
        self.get_logger().info("🎥 Listening for camera streams...")

    def overhead_callback(self, msg):
        try:
            # Convert ROS 2 message to OpenCV format (BGR for normal colors)
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            cv_image = cv2.resize(cv_image, (640, 480))  # Resize for better display
            cv2.imshow("Overhead Camera", cv_image)
            cv2.waitKey(1)
        except Exception as e:
            self.get_logger().error(f"Error converting overhead frame: {e}")

    def gripper_callback(self, msg):
        try:
            # Convert ROS 2 message to OpenCV format
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            cv_image = cv2.resize(cv_image, (640, 480))  # Resize for better display
            cv2.imshow("Gripper Camera", cv_image)
            cv2.waitKey(1)
        except Exception as e:
            self.get_logger().error(f"Error converting gripper frame: {e}")

def main(args=None):
    rclpy.init(args=args)
    node = CameraViewerNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down...")
        
    cv2.destroyAllWindows()
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()