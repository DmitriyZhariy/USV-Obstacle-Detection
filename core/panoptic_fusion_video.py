import cv2
import numpy as np
import torch
import time
from ultralytics import YOLO
import segmentation_models_pytorch as smp
import torch.nn as nn


# =========================
# Загрузка моделей
# =========================
yolo = YOLO(
    r"C:\Users\follo\Downloads\best (5).pt"
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

deeplab = smp.DeepLabV3Plus(
    encoder_name="resnet50",
    encoder_weights=None,
    in_channels=3,
    classes=3
)

deeplab.load_state_dict(torch.load(
    r"E:\Education\4 course 2 semester\Diploma\panoptic_project\runs\semantic_segmentation\deeplab_30\2\deeplab_lars_30.pth",
    map_location=device
))


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
    
# deeplab = UNet(n_classes=3).to(device)
# deeplab.load_state_dict(torch.load(r"E:\Education\4 course 2 semester\Diploma\panoptic_project\runs\semantic_segmentation\UNet\unet_lars_best.pth"))
# deeplab = deeplab.cuda()
# deeplab.eval()


deeplab.to(device)
deeplab.eval()


# =========================
# Цвета
# =========================
YOLO_COLORS = {
    0: (255, 255, 0),
    1: (0, 255, 0),
    2: (255, 0, 255),
    # 3: (255, 165, 0),
    # 4: (255, 69, 0),
    # 5: (255, 192, 203),
    # 6: (255, 215, 0),
    # 7: (139, 0, 255),
}

DL_COLORS = {
    0: (0, 0, 255),
    1: (255, 0, 0),
    2: (0, 255, 255),
}


# =========================
# Препроцессинг
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
# Обработка кадра
# =========================
def process_frame(frame):
    h, w, _ = frame.shape
    frame_draw = frame.copy()

    results = yolo.track(
        frame,
        persist=True,
        tracker="bytetrack.yaml",
        verbose=False
    )[0]

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

    if results.boxes is not None and results.boxes.id is not None:
        boxes = results.boxes
        ids = boxes.id.cpu().numpy().astype(int)
        cls = boxes.cls.cpu().numpy().astype(int)

        masks = results.masks.data.cpu().numpy() if results.masks is not None else None

        for i in range(len(ids)):
            track_id = ids[i]
            class_id = cls[i]

            color = YOLO_COLORS.get(class_id, (255, 255, 255))
            x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy().astype(int)

            cv2.rectangle(frame_draw, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                frame_draw,
                f"{class_id}:{track_id}",
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

    overlay = cv2.addWeighted(frame_draw, 0.7 , semantic_vis, 0.2, 0)
    overlay = cv2.addWeighted(overlay, 0.7, yolo_vis, 0.3, 0)

    # cv2.putText(
    #     overlay,
    #     "Fusion",
    #     (10, 30),
    #     cv2.FONT_HERSHEY_SIMPLEX,
    #     1,
    #     (255, 255, 255),
    #     2,
    #     cv2.LINE_AA
    # )

    return overlay


# =========================
# Видео
# =========================
video_path = r"C:\Users\follo\Videos\Записи экрана\Запись экрана 2026-06-20 034942.mp4"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    raise RuntimeError(f"Не удалось открыть видео: {video_path}")

source_fps = cap.get(cv2.CAP_PROP_FPS)
if source_fps is None or source_fps <= 0:
    source_fps = 30.0

target_fps = 5.0
stride = max(1, int(round(source_fps / target_fps)))
display_delay = int(1000 / target_fps)

print(f"Source FPS: {source_fps:.2f}")
print(f"Target FPS: {target_fps:.2f}")
print(f"Stride: {stride}")

frame_idx = 0
fps_smooth = 0.0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_idx += 1

    # Берём только каждый N-й кадр для обработки
    if (frame_idx - 1) % stride != 0:
        continue

    t0 = time.time()
    overlay = process_frame(frame)
    processing_time = time.time() - t0
    actual_fps = 1.0 / max(processing_time, 1e-6)
    fps_smooth = actual_fps if fps_smooth == 0 else (0.9 * fps_smooth + 0.1 * actual_fps)

    # info1 = f"Video FPS: {source_fps:.1f}"
    # info2 = f"Target FPS: {target_fps:.1f}"
    # info3 = f"Stride: {stride}"
    info4 = f"Actual FPS: {fps_smooth:.1f}"

    # cv2.rectangle(overlay, (10, 40), (280, 150), (0, 0, 0), -1)
    # cv2.putText(overlay, info1, (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
    # cv2.putText(overlay, info2, (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
    # cv2.putText(overlay, info3, (20, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(overlay, info4, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2, cv2.LINE_AA)

    cv2.imshow("Panoptic Viewer", overlay)

    # Если обработка быстрее 200 мс — ждём остаток до 5 FPS
    remaining = display_delay - int(processing_time * 1000)
    delay = max(1, remaining)

    if cv2.waitKey(delay) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()