import os
import io
import json
import argparse
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
from PIL import Image
from tqdm import tqdm


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def detect_columns(df):
    """
    自动检测 image 列和 label 列
    """
    cols = set(df.columns)

    image_candidates = ["image", "img", "jpg", "jpeg", "png"]
    label_candidates = ["label", "labels", "cls", "class", "fine_label"]

    image_col = None
    label_col = None

    for c in image_candidates:
        if c in cols:
            image_col = c
            break

    for c in label_candidates:
        if c in cols:
            label_col = c
            break

    if image_col is None:
        raise ValueError(f"找不到图片列，当前列名: {list(df.columns)}")

    if label_col is None:
        raise ValueError(f"找不到标签列，当前列名: {list(df.columns)}")

    return image_col, label_col


def save_image_from_value(img_value, save_path):
    """
    支持多种 parquet 里图片存储格式：
    1. bytes
    2. dict: {"bytes": ...} / {"path": ...}
    3. PIL Image
    """
    try:
        if isinstance(img_value, dict):
            if "bytes" in img_value and img_value["bytes"] is not None:
                img = Image.open(io.BytesIO(img_value["bytes"])).convert("RGB")
                img.save(save_path, format="JPEG", quality=95)
                return
            elif "path" in img_value and img_value["path"] is not None:
                img = Image.open(img_value["path"]).convert("RGB")
                img.save(save_path, format="JPEG", quality=95)
                return
            else:
                raise ValueError(f"不支持的 dict image 格式: {img_value.keys()}")

        elif isinstance(img_value, bytes):
            img = Image.open(io.BytesIO(img_value)).convert("RGB")
            img.save(save_path, format="JPEG", quality=95)
            return

        elif isinstance(img_value, Image.Image):
            img_value.convert("RGB").save(save_path, format="JPEG", quality=95)
            return

        else:
            # 有些情况下 parquet 读出来是类似 numpy/object，尝试直接用 PIL 打开
            if hasattr(img_value, "get") and callable(img_value.get):
                if img_value.get("bytes", None) is not None:
                    img = Image.open(io.BytesIO(img_value["bytes"])).convert("RGB")
                    img.save(save_path, format="JPEG", quality=95)
                    return

            raise ValueError(f"未知图片类型: {type(img_value)}")
    except Exception as e:
        raise RuntimeError(f"保存图片失败: {save_path}, err={e}")


def process_parquet_files(parquet_files, output_root, split_name, class_map=None, limit=None):
    """
    把一组 parquet 文件处理成 ImageFolder 结构
    """
    split_dir = Path(output_root) / split_name
    ensure_dir(split_dir)

    total_saved = 0
    total_skipped = 0

    for parquet_path in tqdm(parquet_files, desc=f"Processing {split_name} parquet files"):
        table = pq.read_table(parquet_path)
        df = table.to_pandas()

        if len(df) == 0:
            continue

        image_col, label_col = detect_columns(df)

        # 可选：一些数据集可能有 file_name 列
        filename_col = "file_name" if "file_name" in df.columns else None

        for idx, row in tqdm(df.iterrows(), total=len(df), leave=False, desc=f"{os.path.basename(parquet_path)}"):
            if limit is not None and total_saved >= limit:
                print(f"达到 limit={limit}，停止")
                return total_saved, total_skipped

            try:
                label = row[label_col]

                # 类别名：优先用 class_map，否则用数字标签
                class_name = class_map.get(str(label), str(label)) if class_map else str(label)
                class_dir = split_dir / class_name
                ensure_dir(class_dir)

                if filename_col is not None and pd.notna(row[filename_col]):
                    base_name = str(row[filename_col])
                    if "." in base_name:
                        base_name = os.path.splitext(base_name)[0]
                else:
                    base_name = f"{Path(parquet_path).stem}_{idx}"

                save_path = class_dir / f"{base_name}.jpg"

                save_image_from_value(row[image_col], save_path)
                total_saved += 1

            except Exception as e:
                total_skipped += 1
                print(f"[WARN] 跳过样本: parquet={parquet_path}, idx={idx}, err={e}")

    return total_saved, total_skipped


def collect_parquet_files(data_dir, prefix):
    data_dir = Path(data_dir)
    files = sorted(data_dir.glob(f"{prefix}-*.parquet"))
    return [str(f) for f in files]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True,
                        help="包含 train-xxxxx.parquet / validation-xxxxx.parquet 的目录")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="输出 ImageFolder 格式目录")
    parser.add_argument("--class_map_json", type=str, default=None,
                        help="可选，label->class_name 的 json 文件")
    parser.add_argument("--limit", type=int, default=None,
                        help="调试用，只处理前 N 张图")
    args = parser.parse_args()

    class_map = None
    if args.class_map_json is not None:
        with open(args.class_map_json, "r", encoding="utf-8") as f:
            class_map = json.load(f)

    train_files = collect_parquet_files(args.input_dir, "train")
    val_files = collect_parquet_files(args.input_dir, "validation")

    if not train_files:
        print("[WARN] 没找到 train-*.parquet")
    if not val_files:
        print("[WARN] 没找到 validation-*.parquet")

    print(f"找到 train parquet: {len(train_files)} 个")
    print(f"找到 validation parquet: {len(val_files)} 个")

    train_saved, train_skipped = process_parquet_files(
        train_files, args.output_dir, "train", class_map=class_map, limit=args.limit
    )
    print(f"train 完成: saved={train_saved}, skipped={train_skipped}")

    val_saved, val_skipped = process_parquet_files(
        val_files, args.output_dir, "val", class_map=class_map, limit=args.limit
    )
    print(f"val 完成: saved={val_saved}, skipped={val_skipped}")

    print("\n最终目录结构示例：")
    print(f"{args.output_dir}/train/<class_name>/*.jpg")
    print(f"{args.output_dir}/val/<class_name>/*.jpg")


if __name__ == "__main__":
    main()