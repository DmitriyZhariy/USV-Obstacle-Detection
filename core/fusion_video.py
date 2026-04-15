import cv2
import numpy as np
import torch

from ultralytics import YOLO
import segmentation_models_pytorch as smp


yolo = YOLO(r"E:\Education\4 course 2 semester\Practice\panoptic_project\runs\yolo_instance_marine\yolo_640\yolo_final\weights\best.pt")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

deeplab = smp.DeepLabV3Plus(
    encoder_name="resnet50",
    encoder_weights=None,
    in_channels=3,
    classes=3
)

deeplab.load_state_dict(torch.load(r"E:\Education\4 course 2 semester\Practice\panoptic_project\runs\deeplab_semantic_marine\3\deeplab_lars_20.pth"))


deeplab.to(device)
deeplab.eval()


YOLO_COLORS = {
    0: (255,255,0),   
    1: (0,255,0),  
    2: (255,0,255), 
    3: (255,165,0), 
    4: (255,69,0),    
    5: (255,192,203), 
    6: (255,215,0),   
    7: (139,0,255), 
}

DL_COLORS = {
    0: (255, 0, 0),  
    1: (0, 0, 255),     
    2: (0, 255, 255),   
}


def preprocess(frame):
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (512, 512))

    img = img.astype(np.float32) / 255.0

    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    img = (img - mean) / std

    img = torch.from_numpy(img).float()  

    img = img.permute(2, 0, 1).unsqueeze(0)

    return img.to(device)


cap = cv2.VideoCapture(r"E:\Education\4 course 2 semester\Practice\panoptic_project\Video\Запись 2026-04-12 213603.mp4")


INSTANCE_OFFSET = 100

while True:

    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape

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

    panoptic = semantic.copy()

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

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                frame,
                f"{class_id}:{track_id}",
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2
            )

            if masks is not None:

                mask = masks[i]
                mask = (mask > 0.5).astype(np.uint8)

                mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

                panoptic_id = INSTANCE_OFFSET + track_id
                panoptic[mask > 0] = panoptic_id

                yolo_vis[mask > 0] = color

    overlay = cv2.addWeighted(frame, 0.6, semantic_vis, 0.4, 0)
    overlay = cv2.addWeighted(overlay, 0.7, yolo_vis, 0.3, 0)

    cv2.imshow("Panoptic Fusion", overlay)
    cv2.imshow("Semantic", semantic_vis)
    cv2.imshow("Panoptic ID Map", panoptic.astype(np.uint8))

    if cv2.waitKey(1) & 0xFF == 27:
        break


cap.release()
cv2.destroyAllWindows()