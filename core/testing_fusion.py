import cv2
import numpy as np
import torch
import torch.nn as nn
import time
from ultralytics import YOLO
import segmentation_models_pytorch as smp


# =========================
# Загрузка моделей
# =========================
yolo = YOLO(
    r"C:\Users\follo\Downloads\best (5).pt"
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# class DoubleConv(nn.Module):
#     def __init__(self, in_ch, out_ch):
#         super().__init__()
#         self.net = nn.Sequential(
#             nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
#             nn.BatchNorm2d(out_ch),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
#             nn.BatchNorm2d(out_ch),
#             nn.ReLU(inplace=True),
#         )


#     def forward(self, x):
#         return self.net(x)


# class UNet(nn.Module):
#     def __init__(self, n_classes=3):
#         super().__init__()
#         self.down1 = DoubleConv(3, 32)
#         self.down2 = DoubleConv(32, 64)
#         self.down3 = DoubleConv(64, 128)
#         self.down4 = DoubleConv(128, 256)


#         self.pool = nn.MaxPool2d(2)
#         self.bottleneck = DoubleConv(256, 512)


#         self.up4 = nn.ConvTranspose2d(512, 256, 2, stride=2)
#         self.conv4 = DoubleConv(256 + 256, 256)


#         self.up3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
#         self.conv3 = DoubleConv(128 + 128, 128)


#         self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
#         self.conv2 = DoubleConv(64 + 64, 64)


#         self.up1 = nn.ConvTranspose2d(64, 32, 2, stride=2)
#         self.conv1 = DoubleConv(32 + 32, 32)


#         self.out_conv = nn.Conv2d(32, n_classes, kernel_size=1)


#     def forward(self, x):
#         c1 = self.down1(x)
#         p1 = self.pool(c1)


#         c2 = self.down2(p1)
#         p2 = self.pool(c2)


#         c3 = self.down3(p2)
#         p3 = self.pool(c3)


#         c4 = self.down4(p3)
#         p4 = self.pool(c4)


#         bn = self.bottleneck(p4)


#         u4 = self.up4(bn)
#         u4 = torch.cat([u4, c4], dim=1)
#         c5 = self.conv4(u4)


#         u3 = self.up3(c5)
#         u3 = torch.cat([u3, c3], dim=1)
#         c6 = self.conv3(u3)


#         u2 = self.up2(c6)
#         u2 = torch.cat([u2, c2], dim=1)
#         c7 = self.conv2(u2)


#         u1 = self.up1(c7)
#         u1 = torch.cat([u1, c1], dim=1)
#         c8 = self.conv1(u1)


#         logits = self.out_conv(c8)
#         return logits

# unet = UNet(n_classes=3).to(device)
# unet.load_state_dict(torch.load(r"E:\Education\4 course 2 semester\Diploma\panoptic_project\runs\semantic_segmentation\UNet\unet_lars_best.pth"))
# deeplab = unet.cuda()
# deeplab.eval()


deeplab = smp.DeepLabV3Plus(
    encoder_name="resnet50",
    encoder_weights=None,
    in_channels=3,
    classes=3
)

deeplab.load_state_dict(torch.load(
    r"E:\Education\demo\deeplabv3plus_30.pth",
    map_location=device
))

deeplab.to(device)
deeplab.eval()


# =========================
# Цвета
# =========================
YOLO_COLORS = {
    0: (255, 255, 0),
    1: (0, 255, 0),
    2: (255, 0, 255),
    3: (255, 165, 0),
    4: (255, 69, 0),
    5: (255, 192, 203),
    6: (255, 215, 0),
    7: (139, 0, 255),
}

DL_COLORS = {
    0: (255, 0, 0),
    1: (0, 0, 255),
    2: (0, 255, 255),
}

INSTANCE_OFFSET = 100


# =========================
# Вспомогательные функции
# =========================
def preprocess(frame):
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (640, 640))
    img = img.astype(np.float32) / 255.0

    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img = (img - mean) / std

    img = torch.from_numpy(img).float()
    img = img.permute(2, 0, 1).unsqueeze(0)
    return img.to(device)


# =========================
# Видео
# =========================
cap = cv2.VideoCapture(
    r"C:\Users\follo\Videos\Записи экрана\Запись экрана 2026-06-20 034431.mp4"
)

prev_time = time.time()
fps_smooth = 0.0


while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape
    frame_draw = frame.copy()

    results = yolo(frame, verbose=False)[0]

    yolo_vis = np.zeros_like(frame)

    with torch.no_grad():
        pred = deeplab(preprocess(frame))
        pred = torch.argmax(pred, dim=1)[0].cpu().numpy()

    semantic = cv2.resize(
        pred.astype(np.uint8),
        (w, h),
        interpolation=cv2.INTER_NEAREST
    )

    semantic_vis = np.zeros_like(frame)
    for k, color in DL_COLORS.items():
        semantic_vis[semantic == k] = color

    if results.boxes is not None:
        boxes = results.boxes
        cls = boxes.cls.cpu().numpy().astype(int)
        masks = results.masks.data.cpu().numpy() if results.masks is not None else None

        for i in range(len(cls)):
            class_id = cls[i]
            color = YOLO_COLORS.get(class_id, (255, 255, 255))

            x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy().astype(int)

            cv2.rectangle(frame_draw, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                frame_draw,
                f"{class_id}",
                (x1, max(y1 - 5, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
                cv2.LINE_AA
            )

            if masks is not None:
                mask = masks[i]
                mask = (mask > 0.5).astype(np.uint8)
                mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
                yolo_vis[mask > 0] = color

    overlay = cv2.addWeighted(frame_draw, 0.6, semantic_vis, 0.4, 0)
    overlay = cv2.addWeighted(overlay, 0.7, yolo_vis, 0.3, 0)

    cv2.putText(
        overlay, "Fusion", (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA
    )

    # =========================
    # FPS прямо на overlay
    # =========================
    current_time = time.time()
    fps = 1.0 / max(current_time - prev_time, 1e-6)
    prev_time = current_time

    fps_smooth = fps if fps_smooth == 0 else (0.9 * fps_smooth + 0.1 * fps)
    fps_text = f"FPS: {fps_smooth:.1f}"

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.8
    thickness = 2

    (text_w, text_h), baseline = cv2.getTextSize(
        fps_text, font, font_scale, thickness
    )

    x = overlay.shape[1] - text_w - 15
    y = 30

    cv2.rectangle(
        overlay,
        (x - 10, y - text_h - 10),
        (x + text_w + 10, y + baseline + 10),
        (0, 0, 0),
        -1
    )

    cv2.putText(
        overlay,
        fps_text,
        (x, y),
        font,
        font_scale,
        (0, 255, 0),
        thickness,
        cv2.LINE_AA
    )

    cv2.imshow("Panoptic Viewer", overlay)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()