import os
import sys
import glob
import cv2
import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
WEIGHTS = os.path.join(REPO_ROOT, "taskb_perception", "runs", "train_fast", "head_det", "weights", "best.pt")
TEST_DIR = os.path.join(REPO_ROOT, "taskb_perception", "test")
CONF = 0.35
IMGSZ = 640
CLASS_NAMES = {0: "sugar", 1: "mustard", 2: "banana"}
COLORS = {"sugar": (0, 255, 0), "mustard": (0, 200, 255), "banana": (0, 220, 255), "unknown": (128, 255, 0)}

def main():
    from ultralytics import YOLO
    model = YOLO(WEIGHTS)
    print(f"[INFO] model: {WEIGHTS}")
    print(f"[INFO] conf={CONF} imgsz={IMGSZ} names={getattr(model, 'names', None)}")

    exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp")
    paths = []
    for e in exts:
        paths.extend(glob.glob(os.path.join(TEST_DIR, e)))
    paths = sorted(set(paths))
    if not paths:
        print(f"[WARN] no images in {TEST_DIR}")
        return

    for p in paths:
        img_bgr = cv2.imread(p)
        if img_bgr is None:
            print(f"[SKIP] cannot read: {p}")
            continue
        results = model.predict(img_bgr, verbose=False, conf=CONF, imgsz=IMGSZ)

        n_det = 0
        for r in results:
            if r.boxes is None:
                continue
            for box in r.boxes:
                cls_id = int(box.cls.item())
                conf = float(box.conf.item())
                x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
                cls_name = CLASS_NAMES.get(cls_id, "unknown")
                color = COLORS.get(cls_name, (0, 255, 0))
                cv2.rectangle(img_bgr, (x1, y1), (x2, y2), color, 2)
                label = f"{cls_name} {conf:.2f}"
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(img_bgr, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
                cv2.putText(img_bgr, label, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (20, 20, 20), 1, cv2.LINE_AA)
                n_det += 1

        fname = os.path.splitext(os.path.basename(p))[0]
        out_path = os.path.join(TEST_DIR, f"{fname}_det.jpg")
        cv2.imwrite(out_path, img_bgr)
        print(f"[OK] {os.path.basename(p)} -> {os.path.basename(out_path)}  det={n_det}")

    print(f"\n[DONE] output in {TEST_DIR}")

if __name__ == "__main__":
    main()