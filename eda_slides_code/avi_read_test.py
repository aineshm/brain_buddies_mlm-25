import cv2
import xml.etree.ElementTree as ET
import numpy as np

# -----------------------------
# 1. Paths
# -----------------------------
avi_path = "training.avi"       # your AVI file
xml_path = "annotations.xml"    # your XML file

# -----------------------------
# 2. Parse XML
# -----------------------------
tree = ET.parse(xml_path)
root = tree.getroot()

# Dictionaries to store annotations per frame
frames_boxes = {}
frames_polygons = {}
frames_ellipses = {}

for image in root.findall("image"):
    frame_id = int(image.get("id"))

    # Boxes
    for box in image.findall("box"):
        frames_boxes.setdefault(frame_id, []).append({
            "label": box.get("label"),
            "xtl": float(box.get("xtl")),
            "ytl": float(box.get("ytl")),
            "xbr": float(box.get("xbr")),
            "ybr": float(box.get("ybr"))
        })

    # Polygons
    for poly in image.findall("polygon"):
        points = [tuple(map(float, p.split(','))) for p in poly.get("points").split(';')]
        frames_polygons.setdefault(frame_id, []).append({
            "label": poly.get("label"),
            "points": points
        })

    # Ellipses (CVAT exports as <ellipse>)
    for ell in image.findall("ellipse"):
        frames_ellipses.setdefault(frame_id, []).append({
            "label": ell.get("label"),
            "cx": float(ell.get("cx")),
            "cy": float(ell.get("cy")),
            "rx": float(ell.get("rx")),
            "ry": float(ell.get("ry")),
            "angle": float(ell.get("angle"))  # degrees
        })

print(f"Frames with boxes: {sorted(frames_boxes.keys())}")
print(f"Frames with polygons: {sorted(frames_polygons.keys())}")
print(f"Frames with ellipses: {sorted(frames_ellipses.keys())}")

# -----------------------------
# 3. Open video
# -----------------------------
cap = cv2.VideoCapture(avi_path)
frame_id = 0

if not cap.isOpened():
    raise RuntimeError(f"Cannot open video {avi_path}")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # -----------------------------
    # Draw boxes
    # -----------------------------
    for b in frames_boxes.get(frame_id, []):
        cv2.rectangle(frame,
                      (int(b["xtl"]), int(b["ytl"])),
                      (int(b["xbr"]), int(b["ybr"])),
                      (0, 255, 0), 2)
        cv2.putText(frame, b["label"], (int(b["xtl"]), int(b["ytl"])-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # -----------------------------
    # Draw polygons
    # -----------------------------
    for p in frames_polygons.get(frame_id, []):
        pts = np.array(p["points"], np.int32)
        pts = pts.reshape((-1,1,2))
        cv2.polylines(frame, [pts], isClosed=True, color=(255,0,0), thickness=2)
        cv2.putText(frame, p["label"], (int(p["points"][0][0]), int(p["points"][0][1])-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 1)

    # -----------------------------
    # Draw ellipses
    # -----------------------------
    for e in frames_ellipses.get(frame_id, []):
        center = (int(e["cx"]), int(e["cy"]))
        axes = (int(e["rx"]), int(e["ry"]))
        angle = e["angle"]
        cv2.ellipse(frame, center, axes, angle, 0, 360, (0,0,255), 2)
        cv2.putText(frame, e["label"], (int(e["cx"]-e["rx"]), int(e["cy"]-e["ry"])-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)

    # -----------------------------
    # Show frame
    # -----------------------------
    cv2.imshow("Video with Annotations", frame)
    key = cv2.waitKey(30) & 0xFF
    if key == ord("q"):  # press q to quit
        break

    frame_id += 1

cap.release()
cv2.destroyAllWindows()
