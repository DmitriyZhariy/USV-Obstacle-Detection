import os
import cv2
import numpy as np
import xml.etree.ElementTree as ET

XML_PATH = r"C:\Users\User\Downloads\job_6_dataset_2026_05_11_16_42_37_cvat for images 1.1\annotations.xml"
IMAGES_ROOT = r"C:\Users\User\Downloads\job_6_dataset_2026_05_11_16_42_37_cvat for images 1.1\images\right_MOVI0017_0001"
SUBFOLDER_PREFIX = "right_MOVI0017_0001"
OUTPUT_VIDEO = "annotated.mp4"
OPACITY = 0.3

LABEL_COLORS = {
    "Sky": (235, 206, 135),
    "Water": (255, 144, 30),
    "Land": (34, 139, 34),
    "Bridge": (144, 128, 112),
    "Pier": (19, 69, 139),
    "Vessel": (0, 69, 255),
    "Buoy": (0, 215, 255),
    "LandingMark": (255, 0, 255),
    "BridgeLight": (0, 255, 0),
    "Other": (255, 255, 255),
    "Void": (0, 0, 0),
}

def parse_points(points_str):
    pts = []
    for pair in points_str.split(";"):
        pair = pair.strip()
        if not pair:
            continue
        x_str, y_str = pair.split(",")
        pts.append([float(x_str), float(y_str)])
    return np.array(pts, dtype=np.int32)

tree = ET.parse(XML_PATH)
root = tree.getroot()

images_info = []
for image_tag in root.findall("image"):
    name = image_tag.get("name")
    if SUBFOLDER_PREFIX and not name.startswith(SUBFOLDER_PREFIX):
        continue
    width = int(image_tag.get("width"))
    height = int(image_tag.get("height"))
    polygons = []

    for poly in image_tag.findall("polygon"):
        label = poly.get("label")
        points = parse_points(poly.get("points"))
        z_order = int(poly.get("z_order", "0"))
        polygons.append((z_order, label, points))

    polygons.sort(key=lambda x: x[0])
    images_info.append((name, width, height, polygons))

def frame_index(name):
    base = os.path.basename(name)
    stem, _ = os.path.splitext(base)
    try:
        return int(stem)
    except ValueError:
        return stem

images_info.sort(key=lambda x: frame_index(x[0]))

if not images_info:
    raise RuntimeError("No images found in XML")

first_img_path = os.path.join(IMAGES_ROOT, os.path.basename(images_info[0][0]))
frame0 = cv2.imread(first_img_path)
if frame0 is None:
    raise RuntimeError(f"Cannot read first image at {first_img_path}")

h, w = frame0.shape[:2]
fps = 5.0

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
writer = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (w, h))

for name, width, height, polygons in images_info:
    img_path = os.path.join(IMAGES_ROOT, os.path.basename(name))
    frame = cv2.imread(img_path)
    if frame is None:
        print(f"Warning: cannot read {img_path}, skipping")
        continue

    overlay = frame.copy()
    for _, label, pts in polygons:
        color = LABEL_COLORS.get(label, (0, 255, 0))
        cv2.fillPoly(overlay, [pts], color)
        cv2.polylines(overlay, [pts], isClosed=True, color=(0, 0, 0), thickness=1)
    cv2.addWeighted(overlay, OPACITY, frame, 1 - OPACITY, 0, frame)

    writer.write(frame)

writer.release()
print("Saved", OUTPUT_VIDEO)
