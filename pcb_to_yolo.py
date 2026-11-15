"""
Convert a COCO-format object detection dataset (e.g. Mendeley PCB-Defect)
into the YOLO layout expected by yolov9 / ultralytics:

    out_dir/
      images/train, images/val, images/test
      labels/train, labels/val, labels/test
      pcb.yaml

Usage (Anaconda Prompt, env yolov9):
    python pcb_to_yolo.py --ann path\\to\\annotations.json --img_dir path\\to\\images --out pcb_yolo

For EdgePCB (already YOLO txt labels), use --yolo_src instead:
    python pcb_to_yolo.py --yolo_src path\\to\\edgepcb --out pcb_yolo
"""
import argparse
import json
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path


def write_yaml(out_dir: Path, names):
    yaml_path = out_dir / "pcb.yaml"
    lines = [
        f"path: {out_dir.resolve().as_posix()}",
        "train: images/train",
        "val: images/val",
        "test: images/test",
        f"nc: {len(names)}",
        "names: [" + ", ".join(f"'{n}'" for n in names) + "]",
    ]
    yaml_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[ok] wrote {yaml_path}")
    return yaml_path


def ensure_dirs(out_dir: Path):
    for split in ("train", "val", "test"):
        (out_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_dir / "labels" / split).mkdir(parents=True, exist_ok=True)


def split_ids(ids, seed, ratios=(0.8, 0.1, 0.1)):
    ids = sorted(ids)
    random.Random(seed).shuffle(ids)
    n = len(ids)
    n_train = int(n * ratios[0])
    n_val = int(n * ratios[1])
    return {
        "train": ids[:n_train],
        "val": ids[n_train:n_train + n_val],
        "test": ids[n_train + n_val:],
    }


def coco_to_yolo(ann_path: Path, img_dir: Path, out_dir: Path, seed: int):
    data = json.loads(ann_path.read_text(encoding="utf-8"))

    # category id -> contiguous index, sorted by original id for stability
    cats = sorted(data["categories"], key=lambda c: c["id"])
    cat_to_idx = {c["id"]: i for i, c in enumerate(cats)}
    names = [c["name"] for c in cats]

    images = {im["id"]: im for im in data["images"]}
    anns_by_img = defaultdict(list)
    for a in data["annotations"]:
        anns_by_img[a["image_id"]].append(a)

    ensure_dirs(out_dir)
    splits = split_ids(images.keys(), seed)

    class_count = Counter()
    skipped_boxes = 0
    missing_files = 0

    for split, ids in splits.items():
        for img_id in ids:
            im = images[img_id]
            src = img_dir / im["file_name"]
            if not src.exists():
                # some datasets keep images in subfolders; try basename search
                matches = list(img_dir.rglob(Path(im["file_name"]).name))
                if not matches:
                    missing_files += 1
                    continue
                src = matches[0]

            w_img, h_img = float(im["width"]), float(im["height"])
            dst_img = out_dir / "images" / split / src.name
            shutil.copy2(src, dst_img)

            lines = []
            for a in anns_by_img.get(img_id, []):
                x, y, w, h = a["bbox"]  # COCO: top-left x, y, width, height
                if w <= 0 or h <= 0:
                    skipped_boxes += 1
                    continue
                # clamp to image bounds
                x1, y1 = max(0.0, x), max(0.0, y)
                x2, y2 = min(w_img, x + w), min(h_img, y + h)
                if x2 <= x1 or y2 <= y1:
                    skipped_boxes += 1
                    continue
                xc = (x1 + x2) / 2 / w_img
                yc = (y1 + y2) / 2 / h_img
                bw = (x2 - x1) / w_img
                bh = (y2 - y1) / h_img
                cls = cat_to_idx[a["category_id"]]
                lines.append(f"{cls} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")
                class_count[names[cls]] += 1

            label_path = out_dir / "labels" / split / (src.stem + ".txt")
            label_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

    yaml_path = write_yaml(out_dir, names)

    print("\n=== summary ===")
    for split, ids in splits.items():
        print(f"{split:5s}: {len(ids)} images (target)")
    print("boxes per class:", dict(class_count))
    print(f"skipped degenerate boxes: {skipped_boxes}")
    print(f"missing image files: {missing_files}")
    return yaml_path


def copy_yolo_source(src: Path, out_dir: Path):
    """For datasets that already ship in YOLO layout (e.g. EdgePCB)."""
    # Expect src/images/{train,val,test} and src/labels/{train,val,test},
    # or src/{train,valid,test}/{images,labels}. Adjust if the download differs.
    layouts = [
        (src / "images", src / "labels"),
    ]
    found = None
    for img_root, lab_root in layouts:
        if img_root.exists() and lab_root.exists():
            found = (img_root, lab_root)
            break

    if found is None:
        # Roboflow-style: train/images, valid/images, test/images
        names_map = {"train": "train", "valid": "val", "val": "val", "test": "test"}
        ensure_dirs(out_dir)
        counts = Counter()
        for src_split, dst_split in names_map.items():
            img_dir = src / src_split / "images"
            lab_dir = src / src_split / "labels"
            if not img_dir.exists():
                continue
            for img in img_dir.iterdir():
                shutil.copy2(img, out_dir / "images" / dst_split / img.name)
                lab = lab_dir / (img.stem + ".txt")
                dst_lab = out_dir / "labels" / dst_split / (img.stem + ".txt")
                if lab.exists():
                    shutil.copy2(lab, dst_lab)
                    counts[dst_split] += 1
                else:
                    dst_lab.write_text("", encoding="utf-8")
        print("copied (roboflow layout):", dict(counts))
    else:
        raise SystemExit("Unsupported layout; inspect the download and adjust copy_yolo_source().")

    # names: read from data.yaml if present in source
    names = []
    for cand in ("data.yaml", "dataset.yaml"):
        p = src / cand
        if p.exists():
            text = p.read_text(encoding="utf-8")
            for line in text.splitlines():
                if line.strip().startswith("names:"):
                    names_raw = line.split("names:", 1)[1].strip()
                    names = [n.strip().strip("'\"") for n in names_raw.strip("[]").split(",") if n.strip()]
            break
    if not names:
        raise SystemExit("Could not read class names; set them manually in pcb.yaml.")
    write_yaml(out_dir, names)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ann", type=Path, help="COCO annotation json")
    ap.add_argument("--img_dir", type=Path, help="folder with the images")
    ap.add_argument("--yolo_src", type=Path, help="YOLO-format dataset root (e.g. EdgePCB)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if args.ann and args.img_dir:
        coco_to_yolo(args.ann, args.img_dir, args.out, args.seed)
    elif args.yolo_src:
        copy_yolo_source(args.yolo_src, args.out)
    else:
        ap.error("provide either --ann + --img_dir, or --yolo_src")


if __name__ == "__main__":
    main()
