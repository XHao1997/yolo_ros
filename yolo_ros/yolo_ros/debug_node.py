# Copyright (C) 2023 Miguel Ángel González Santamarta

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.


import cv2
import random
import numpy as np
from typing import Tuple

import rclpy
from rclpy.duration import Duration
from rclpy.qos import QoSProfile
from rclpy.qos import QoSHistoryPolicy
from rclpy.qos import QoSDurabilityPolicy
from rclpy.qos import QoSReliabilityPolicy
from rclpy.lifecycle import LifecycleNode
from rclpy.lifecycle import TransitionCallbackReturn
from rclpy.lifecycle import LifecycleState

import message_filters
from cv_bridge import CvBridge
from ultralytics.utils.plotting import Annotator, colors

from sensor_msgs.msg import Image
from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray
from yolo_msgs.msg import BoundingBox2D
from yolo_msgs.msg import KeyPoint2D
from yolo_msgs.msg import KeyPoint3D
from yolo_msgs.msg import Detection
from yolo_msgs.msg import DetectionArray


class DebugNode(LifecycleNode):

    def __init__(self) -> None:
        super().__init__("debug_node")

        self._class_to_color = {}
        self.cv_bridge = CvBridge()

        # params
        self.declare_parameter("image_reliability", QoSReliabilityPolicy.BEST_EFFORT)

    def on_configure(self, state: LifecycleState) -> TransitionCallbackReturn:
        self.get_logger().info(f"[{self.get_name()}] Configuring...")

        self.image_qos_profile = QoSProfile(
            reliability=self.get_parameter("image_reliability")
            .get_parameter_value()
            .integer_value,
            history=QoSHistoryPolicy.KEEP_LAST,
            durability=QoSDurabilityPolicy.VOLATILE,
            depth=1,
        )

        # pubs
        self._dbg_pub = self.create_publisher(Image, "dbg_image", 10)
        self._bb_markers_pub = self.create_publisher(MarkerArray, "dgb_bb_markers", 10)
        self._kp_markers_pub = self.create_publisher(MarkerArray, "dgb_kp_markers", 10)

        super().on_configure(state)
        self.get_logger().info(f"[{self.get_name()}] Configured")

        return TransitionCallbackReturn.SUCCESS

    def on_activate(self, state: LifecycleState) -> TransitionCallbackReturn:
        self.get_logger().info(f"[{self.get_name()}] Activating...")

        # subs
        self.image_sub = message_filters.Subscriber(
            self, Image, "image_raw", qos_profile=self.image_qos_profile
        )
        self.detections_sub = message_filters.Subscriber(
            self, DetectionArray, "detections", qos_profile=10
        )

        self._synchronizer = message_filters.ApproximateTimeSynchronizer(
            (self.image_sub, self.detections_sub), 10, 0.5
        )
        self._synchronizer.registerCallback(self.detections_cb)

        super().on_activate(state)
        self.get_logger().info(f"[{self.get_name()}] Activated")

        return TransitionCallbackReturn.SUCCESS

    def on_deactivate(self, state: LifecycleState) -> TransitionCallbackReturn:
        self.get_logger().info(f"[{self.get_name()}] Deactivating...")

        self.destroy_subscription(self.image_sub.sub)
        self.destroy_subscription(self.detections_sub.sub)

        del self._synchronizer

        super().on_deactivate(state)
        self.get_logger().info(f"[{self.get_name()}] Deactivated")

        return TransitionCallbackReturn.SUCCESS

    def on_cleanup(self, state: LifecycleState) -> TransitionCallbackReturn:
        self.get_logger().info(f"[{self.get_name()}] Cleaning up...")

        self.destroy_publisher(self._dbg_pub)
        self.destroy_publisher(self._bb_markers_pub)
        self.destroy_publisher(self._kp_markers_pub)

        super().on_cleanup(state)
        self.get_logger().info(f"[{self.get_name()}] Cleaned up")

        return TransitionCallbackReturn.SUCCESS

    def on_shutdown(self, state: LifecycleState) -> TransitionCallbackReturn:
        self.get_logger().info(f"[{self.get_name()}] Shutting down...")
        super().on_cleanup(state)
        self.get_logger().info(f"[{self.get_name()}] Shutted down")
        return TransitionCallbackReturn.SUCCESS
    
    def _draw_filled_transparent_rect(self, img, pt1, pt2, color, alpha=0.6, border=2, radius=10, shadow=3):
        """
        Fancy rounded pill:
        • soft drop shadow
        • semi-transparent color tint
        • anti-aliased rounded border
        color: BGR tuple
        """
        x1, y1 = map(int, pt1); x2, y2 = map(int, pt2)
        if x2 < x1: x1, x2 = x2, x1
        if y2 < y1: y1, y2 = y2, y1
        H, W = img.shape[:2]
        x1 = max(0, min(W-1, x1)); x2 = max(0, min(W, x2))
        y1 = max(0, min(H-1, y1)); y2 = max(0, min(H, y2))
        if x2 - x1 < 2 or y2 - y1 < 2:
            return

        # --- helpers ---
        def rounded(mask, x1, y1, x2, y2, r, val=255):
            r = int(max(0, min(r, (x2-x1)//2, (y2-y1)//2)))
            if r == 0:
                cv2.rectangle(mask, (x1, y1), (x2-1, y2-1), val, -1)
                return
            cv2.rectangle(mask, (x1+r, y1), (x2-r, y2-1), val, -1)
            cv2.rectangle(mask, (x1, y1+r), (x2-1, y2-r), val, -1)
            cv2.circle(mask, (x1+r, y1+r), r, val, -1)
            cv2.circle(mask, (x2-r-1, y1+r), r, val, -1)
            cv2.circle(mask, (x1+r, y2-r-1), r, val, -1)
            cv2.circle(mask, (x2-r-1, y2-r-1), r, val, -1)

        def blend_mask(base, overlay, mask_u8):
            m = (mask_u8.astype(np.float32) / 255.0)[..., None]  # HxWx1
            return (overlay.astype(np.float32) * m + base.astype(np.float32) * (1.0 - m)).astype(np.uint8)

        # --- shadow (soft, offset) ---
        if shadow and shadow > 0:
            m_shadow = np.zeros((H, W), np.uint8)
            rounded(m_shadow, x1+shadow, y1+shadow, x2+shadow, y2+shadow, radius)
            m_shadow = cv2.GaussianBlur(m_shadow, (0, 0), 5.0)
            dark = (img.astype(np.float32) * 0.7).astype(np.uint8)  # 30% darker under shadow
            img[:] = blend_mask(img, dark, m_shadow)

        # --- main fill (semi-transparent tint inside rounded rect) ---
        mask = np.zeros((H, W), np.uint8)
        rounded(mask, x1, y1, x2, y2, radius)
        # optional feather for softer edges
        mask = cv2.GaussianBlur(mask, (0, 0), 0.8)

        overlay = np.zeros_like(img); overlay[:] = color
        # pre-mix alpha so fill isn’t fully opaque
        mixed = (overlay.astype(np.float32) * alpha + img.astype(np.float32) * (1.0 - alpha)).astype(np.uint8)
        img[:] = blend_mask(img, mixed, mask)

        # --- crisp anti-aliased border following the rounded shape ---
        if border and border > 0:
            cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if cnts:
                # slightly brighter border for pop
                b, g, r = map(int, color)
                border_col = (min(255, int(b*1.2)), min(255, int(g*1.2)), min(255, int(r*1.2)))
                cv2.drawContours(img, cnts, -1, border_col, border, lineType=cv2.LINE_AA)
    def draw_box(
        self,
        cv_image: np.ndarray,
        detection,  # Detection
        color: Tuple[int, int, int],
        ) -> np.ndarray:
        """
        Pretty rotated box with label pill & text outline.
        `color` is BGR (OpenCV).
        """
        class_name = detection.class_name or "obj"
        score = detection.score if detection.score is not None else 0.0
        box_msg = detection.bbox  # BoundingBox2D
        track_id = getattr(detection, "id", None)

        cx = float(box_msg.center.position.x)
        cy = float(box_msg.center.position.y)
        w  = float(box_msg.size.x)
        h  = float(box_msg.size.y)
        ang_deg = -np.degrees(float(getattr(box_msg.center, "theta", 0.0)))  # match your previous sign

        # Build rotated rect via OpenCV (more stable than manual matrix math)
        rect = ((cx, cy), (max(1.0, w), max(1.0, h)), ang_deg)
        box = cv2.boxPoints(rect)              # 4x2 float
        box_i = np.intp(np.round(box))         # int coords

        # ---- draw outline with a soft "shadow" first (contrast on bright bg)
        shadow_col = (0, 0, 0)
        cv2.polylines(cv_image, [box_i], isClosed=True, color=shadow_col, thickness=4, lineType=cv2.LINE_AA)
        cv2.polylines(cv_image, [box_i], isClosed=True, color=color,       thickness=2, lineType=cv2.LINE_AA)

        # ---- draw small corner dots (subtle styling)
        for pt in box_i:
            cv2.circle(cv_image, tuple(pt), 2, color, -1, lineType=cv2.LINE_AA)

        # ---- label text
        id_part = f" ({track_id})" if track_id not in (None, "", -1) else ""
        label = f"{class_name}{id_part}  {score:.3f}"

        # Measure text
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness_text = 1
        (tw, th), baseline = cv2.getTextSize(label, font, font_scale, thickness_text)
        pad = 6

        # Place label above the topmost box point if possible; otherwise inside
        topmost = box_i[np.argmin(box[:, 1])]
        lx = int(topmost[0] - tw // 2)  # center text over topmost x
        ly = int(topmost[1] - 8 - th)   # a bit above the box edge

        # Keep on-screen; if off top, move inside the box
        h_img, w_img = cv_image.shape[:2]
        lx = max(2, min(w_img - tw - 2, lx))
        ly = int(min(box[:, 1]) + th + pad + 2)
        # Background pill (semi-transparent)
        self._draw_filled_transparent_rect(
            cv_image,
            (lx - pad, ly - th - pad),
            (lx + tw + pad, ly + baseline + pad // 2),
            color=color,
            alpha=0.65,
            border=1
        )

        # Text with thin dark outline for readability
        # Outline: draw black text slightly thicker underneath
        cv2.putText(cv_image, label, (lx, ly), font, font_scale, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(cv_image, label, (lx, ly), font, font_scale, (255, 255, 255), 1, cv2.LINE_AA)

        # ---- optional: draw a small center crosshair & heading tick
        cc = (int(round(cx)), int(round(cy)))
        cv2.circle(cv_image, cc, 2, (255, 255, 255), -1, lineType=cv2.LINE_AA)
        # heading tick (10 px from center along the box angle)
        rad = np.radians(-ang_deg)
        tip = (int(round(cx + 10 * np.cos(rad))), int(round(cy + 10 * np.sin(rad))))
        cv2.line(cv_image, cc, tip, (255, 255, 255), 1, lineType=cv2.LINE_AA)

        return cv_image
    def draw_mask(
        self,
        cv_image: np.ndarray,
        detection: Detection,
        color: Tuple[int],
    ) -> np.ndarray:

        mask_msg = detection.mask
        mask_array = np.array([[int(ele.x), int(ele.y)] for ele in mask_msg.data])

        if mask_msg.data:
            layer = cv_image.copy()
            layer = cv2.fillPoly(layer, pts=[mask_array], color=color)
            cv2.addWeighted(cv_image, 0.4, layer, 0.6, 0, cv_image)
            cv_image = cv2.polylines(
                cv_image,
                [mask_array],
                isClosed=True,
                color=color,
                thickness=2,
                lineType=cv2.LINE_AA,
            )
        return cv_image

    def draw_keypoints(self, cv_image: np.ndarray, detection: Detection) -> np.ndarray:

        keypoints_msg = detection.keypoints

        ann = Annotator(cv_image)

        kp: KeyPoint2D
        for kp in keypoints_msg.data:
            color_k = (
                [int(x) for x in ann.kpt_color[kp.id - 1]]
                if len(keypoints_msg.data) == 17
                else colors(kp.id - 1)
            )

            cv2.circle(
                cv_image,
                (int(kp.point.x), int(kp.point.y)),
                5,
                color_k,
                -1,
                lineType=cv2.LINE_AA,
            )
            cv2.putText(
                cv_image,
                str(kp.id),
                (int(kp.point.x), int(kp.point.y)),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                color_k,
                1,
                cv2.LINE_AA,
            )

        def get_pk_pose(kp_id: int) -> Tuple[int]:
            for kp in keypoints_msg.data:
                if kp.id == kp_id:
                    return (int(kp.point.x), int(kp.point.y))
            return None

        for i, sk in enumerate(ann.skeleton):
            kp1_pos = get_pk_pose(sk[0])
            kp2_pos = get_pk_pose(sk[1])

            if kp1_pos is not None and kp2_pos is not None:
                cv2.line(
                    cv_image,
                    kp1_pos,
                    kp2_pos,
                    [int(x) for x in ann.limb_color[i]],
                    thickness=2,
                    lineType=cv2.LINE_AA,
                )

        return cv_image

    def create_bb_marker(self, detection: Detection, color: Tuple[int]) -> Marker:

        bbox3d = detection.bbox3d

        marker = Marker()
        marker.header.frame_id = bbox3d.frame_id

        marker.ns = "yolo_3d"
        marker.type = Marker.CUBE
        marker.action = Marker.ADD
        marker.frame_locked = False

        marker.pose.position.x = bbox3d.center.position.x
        marker.pose.position.y = bbox3d.center.position.y
        marker.pose.position.z = bbox3d.center.position.z

        marker.pose.orientation.x = 0.0
        marker.pose.orientation.y = 0.0
        marker.pose.orientation.z = 0.0
        marker.pose.orientation.w = 1.0
        marker.scale.x = bbox3d.size.x
        marker.scale.y = bbox3d.size.y
        marker.scale.z = bbox3d.size.z

        marker.color.r = color[0] / 255.0
        marker.color.g = color[1] / 255.0
        marker.color.b = color[2] / 255.0
        marker.color.a = 0.4

        marker.lifetime = Duration(seconds=0.5).to_msg()
        marker.text = detection.class_name

        return marker

    def create_kp_marker(self, keypoint: KeyPoint3D) -> Marker:

        marker = Marker()

        marker.ns = "yolo_3d"
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.frame_locked = False

        marker.pose.position.x = keypoint.point.x
        marker.pose.position.y = keypoint.point.y
        marker.pose.position.z = keypoint.point.z

        marker.pose.orientation.x = 0.0
        marker.pose.orientation.y = 0.0
        marker.pose.orientation.z = 0.0
        marker.pose.orientation.w = 1.0
        marker.scale.x = 0.05
        marker.scale.y = 0.05
        marker.scale.z = 0.05

        marker.color.r = (1.0 - keypoint.score) * 255.0
        marker.color.g = 0.0
        marker.color.b = keypoint.score * 255.0
        marker.color.a = 0.4

        marker.lifetime = Duration(seconds=0.5).to_msg()
        marker.text = str(keypoint.id)

        return marker

    def detections_cb(self, img_msg: Image, detection_msg: DetectionArray) -> None:
        cv_image = self.cv_bridge.imgmsg_to_cv2(img_msg, desired_encoding="bgr8")
        bb_marker_array = MarkerArray()
        kp_marker_array = MarkerArray()

        detection: Detection
        for detection in detection_msg.detections:

            # random color
            class_name = detection.class_name

            if class_name not in self._class_to_color:
                r = random.randint(0, 255)
                g = random.randint(0, 255)
                b = random.randint(0, 255)
                self._class_to_color[class_name] = (r, g, b)

            color = self._class_to_color[class_name]

            cv_image = self.draw_box(cv_image, detection, color)
            cv_image = self.draw_mask(cv_image, detection, color)
            cv_image = self.draw_keypoints(cv_image, detection)

            if detection.bbox3d.frame_id:
                marker = self.create_bb_marker(detection, color)
                marker.header.stamp = img_msg.header.stamp
                marker.id = len(bb_marker_array.markers)
                bb_marker_array.markers.append(marker)

            if detection.keypoints3d.frame_id:
                for kp in detection.keypoints3d.data:
                    marker = self.create_kp_marker(kp)
                    marker.header.frame_id = detection.keypoints3d.frame_id
                    marker.header.stamp = img_msg.header.stamp
                    marker.id = len(kp_marker_array.markers)
                    kp_marker_array.markers.append(marker)

        # publish dbg image
        self._dbg_pub.publish(
            self.cv_bridge.cv2_to_imgmsg(cv_image, encoding="bgr8", header=img_msg.header)
        )
        self._bb_markers_pub.publish(bb_marker_array)
        self._kp_markers_pub.publish(kp_marker_array)


def main():
    rclpy.init()
    node = DebugNode()
    node.trigger_configure()
    node.trigger_activate()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
