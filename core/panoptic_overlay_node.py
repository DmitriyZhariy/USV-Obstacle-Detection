import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

import cv2
import numpy as np
import torch

from ultralytics import YOLO
import segmentation_models_pytorch as smp


class PanopticNode(Node):

    def __init__(self):
        super().__init__('panoptic_node')

        self.bridge = CvBridge()

        # ---------------- ROS ----------------
        self.sub = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.callback,
            10
        )

        self.pub = self.create_publisher(
            Image,
            '/panoptic/image_raw',
            10
        )

        self.panoptic_pub = self.create_publisher(
            Image,
            '/panoptic/id_map',
            10
        )

        # ---------------- YOLO ----------------
        self.yolo = YOLO(r"E:\Education\4 course 2 semester\Practice\panoptic_project\runs\yolo_instance_marine\yolo_resized\yolo_40_832\weights\best.pt")

        # ---------------- DeepLab ----------------
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.deeplab = smp.DeepLabV3Plus(
            encoder_name="resnet50",
            encoder_weights=None,
            in_channels=3,
            classes=3
        )

        self.deeplab.load_state_dict(
            torch.load(r"E:\Education\4 course 2 semester\Practice\panoptic_project\runs\deeplab_semantic_marine\3\deeplab_lars_20.pth", map_location=self.device)
        )

        self.deeplab.to(self.device)
        self.deeplab.eval()

        # ---------------- COLORS ----------------
        self.yolo_colors = self._build_yolo_colors()
        self.dl_colors = self._build_dl_colors()

    def _build_yolo_colors(self):
        return {
            0: (255, 0, 0),
            1: (0, 255, 0),
            2: (0, 0, 255),
            3: (255, 255, 0),
            4: (255, 0, 255),
            5: (0, 255, 255),
            6: (128, 0, 128),
            7: (255, 128, 0)
        }

    def _build_dl_colors(self):
        return {
            0: (0, 0, 255),
            1: (135, 206, 235),
            2: (0, 255, 0)
        }

    def preprocess(self, frame):
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (512, 512))

        img = img.astype(np.float32) / 255.0

        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])

        img = (img - mean) / std

        img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(self.device)

        return img

    # ================= CALLBACK =================
    def callback(self, msg):

        frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        h, w, _ = frame.shape

        # ======================================================
        # YOLO + TRACK
        # ======================================================
        results = self.yolo.track(
            frame,
            persist=True,
            tracker="bytetrack.yaml",
            verbose=False
        )

        results = results[0] if isinstance(results, list) else results

        yolo_vis = np.zeros_like(frame)

        # ======================================================
        # DEEPLAB
        # ======================================================
        with torch.no_grad():
            pred = self.deeplab(self.preprocess(frame))
            pred = torch.argmax(pred, dim=1)[0].cpu().numpy()

        semantic = cv2.resize(pred.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)

        semantic_vis = np.zeros_like(frame)

        for k, v in self.dl_colors.items():
            semantic_vis[semantic == k] = v

        # ======================================================
        # PANOPTIC MAP
        # ======================================================
        panoptic = semantic.copy()
        INSTANCE_OFFSET = 100

        # ======================================================
        # YOLO INSTANCES
        # ======================================================
        if results.boxes is not None and results.boxes.id is not None:

            boxes = results.boxes
            ids = boxes.id.cpu().numpy().astype(int)
            cls = boxes.cls.cpu().numpy().astype(int)

            masks = results.masks.data.cpu().numpy() if results.masks is not None else None

            for i in range(len(ids)):

                track_id = ids[i]
                class_id = cls[i]

                color = self.yolo_colors.get(class_id, (255, 255, 255))

                x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy().astype(int)

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"{class_id}:{track_id}",
                            (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            color,
                            2)

                if masks is not None:

                    mask = masks[i]
                    mask = (mask > 0.5).astype(np.uint8)

                    mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

                    panoptic_id = INSTANCE_OFFSET + track_id
                    panoptic[mask > 0] = panoptic_id

                    yolo_vis[mask > 0] = color

        # ======================================================
        # OUTPUT VISUALIZATION
        # ======================================================
        overlay = cv2.addWeighted(frame, 0.6, semantic_vis, 0.4, 0)
        overlay = cv2.addWeighted(overlay, 0.7, yolo_vis, 0.3, 0)

        # ======================================================
        # PUBLISH
        # ======================================================
        self.pub.publish(self.bridge.cv2_to_imgmsg(overlay, 'bgr8'))

        self.panoptic_pub.publish(
            self.bridge.cv2_to_imgmsg(panoptic.astype(np.uint8), 'mono8')
        )


def main():
    rclpy.init()
    node = PanopticNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()