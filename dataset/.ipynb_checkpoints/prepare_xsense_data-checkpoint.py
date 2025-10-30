#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
prepare_xsense_with_cmd.py  (run-and-go, revised)

功能：
- 从 /data/LyuShipeng/Xsense_dataset/Xsense_dataset 构建"一一对应"样本
- 输入：区间帧 difference 图 + contact_depth.png + mass.txt + texture.txt + cmd_mm
- 输出：best 帧 difference 图
- 完全整体打乱数据，确保测试集包含各个组别的代表性样本
- 默认比例 0.8/0.19/0.01
- 使用固定种子确保结果可重复
- 根据PATH_MODE设置保存文件（相对路径/绝对路径/符号链接/复制文件）
"""

import os, re, csv, sys, json, random, shutil, statistics
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict

# ========================= 配置 =========================
XSENSE_ROOT     = "/data/LyuShipeng/Xsense_dataset/Xsense_dataset"
OUT_DIR         = "/code/Xense_tactile/dataset"

TRAIN_RATIO     = 0.975
VAL_RATIO       = 0.015
TEST_RATIO      = 0.01
SEED            = 42

NORMALIZE       = True
END_INCLUSIVE   = True
VERIFY_IMAGES   = True
SKIP_BROKEN     = True
PATH_MODE       = "relative"  # 可选: "relative", "absolute", "symlink", "copy"
IMG_EXTS        = [".png",".jpg",".jpeg",".bmp",".tif",".tiff"]

TEXTURE_MAP     = None
GROUP_REGEX     = None

INDEX_OFFSET    = 0
EMIT_FB         = False
EMIT_TS         = False
EMIT_CSV        = False
# ========================================================

try:
    from PIL import Image
except Exception:
    Image = None

# ---------------------------- I/O ----------------------------
def read_text(p: Path) -> str:
    try: return p.read_text(encoding="utf-8").strip()
    except Exception: return p.read_text(errors="ignore").strip()

# ---------------------------- mass/texture 规范化 ----------------------------
_JIN_TO_G = 500.0
_OZ_TO_G  = 28.349523125
_LB_TO_G  = 453.59237

def normalize_mass(s: str) -> str:
    if not s: return s
    raw = s.strip()
    repl = (("千克","kg"),("公斤","kg"),("克","g"),("毫克","mg"),("斤","jin"),("两","liang"))
    for a,b in repl: raw = raw.replace(a,b)
    m = re.match(r"^\s*([0-9]+(?:\.[0-9]+)?)\s*([a-zA-Z]+)?", raw)
    if not m: return raw
    val = float(m.group(1)); unit = (m.group(2) or "g").lower()
    if unit in ("g","gram","grams"): g = val
    elif unit in ("kg","kilogram","kilograms"): g = val*1000
    elif unit in ("mg",): g = val/1000
    elif unit in ("lb","lbs","pound","pounds"): g = val*_LB_TO_G
    elif unit in ("oz","ounce","ounces"): g = val*_OZ_TO_G
    elif unit in ("jin",): g = val*_JIN_TO_G
    elif unit in ("liang",): g = val*_JIN_TO_G/16.0
    else: return raw
    return f"{g:g} g"

def normalize_texture(s: str) -> str:
    if s is None: return s
    return re.sub(r"\s+"," ", s).strip().lower()

# ---------------------------- 工具函数 ----------------------------
def verify_image_ok(p: Path):
    if Image is None: return True, None
    try:
        with Image.open(p) as im: im.verify()
        with Image.open(p) as im2: sz = im2.size
        return True, sz
    except Exception:
        return False, None

def maybe_materialize(p: Path, root: Path, out_root: Path, mode: str, subdir_hint=None) -> str:
    if mode=="relative":
        try: return p.relative_to(root).as_posix()
        except: return p.as_posix()
    if mode=="absolute": return p.as_posix()
    sub = subdir_hint or p.parent.name
    dst_dir = out_root / sub; dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / p.name
    if mode=="symlink":
        if dst.exists() or dst.is_symlink(): dst.unlink()
        os.symlink(p.as_posix(), dst.as_posix())
    elif mode=="copy":
        if not dst.exists(): shutil.copy2(p, dst)
    return dst.relative_to(out_root).as_posix()

def parse_three_ints(p: Path):
    nums = list(map(int, re.findall(r"-?\d+", read_text(p))))
    if len(nums) < 3: raise ValueError(f"F.txt 格式错误: {p}")
    return nums[0], nums[1], nums[2]

def map_frame_files(diff_dir: Path, img_exts: set):
    mapping = {}
    for fp in diff_dir.iterdir():
        if fp.is_file() and fp.suffix.lower() in img_exts:
            m = re.search(r"(\d+)", fp.stem)
            if m: mapping[int(m.group(1))] = fp
    return mapping

# ---------------------------- gripper_log ----------------------------
def read_gripper_log(csv_path: Path):
    if not csv_path.exists(): return {'rows': [], 'has_frame_col': False}
    rows=[]; has_frame=False
    with csv_path.open('r', encoding='utf-8') as f:
        reader=csv.DictReader(f)
        lower=[k.lower() for k in (reader.fieldnames or [])]
        if 'frame' in lower: has_frame=True
        for i,row in enumerate(reader):
            def tf(x): 
                try: return float(x)
                except: return None
            def ti(x):
                try: return int(float(x))
                except: return None
            rows.append({
                'timestamp_ms': ti(row.get('timestamp_ms')),
                'cmd_mm': tf(row.get('cmd_mm')),
                'fb_mm': tf(row.get('fb_mm')),
                'frame': ti(row.get('frame')) if has_frame else None,
                '_row_index': i
            })
    return {'rows': rows, 'has_frame_col': has_frame}

def get_cmd_by_frame(grip, frame_idx):
    target=None
    if grip['rows']:
        if grip['has_frame_col']:
            wanted = frame_idx + INDEX_OFFSET
            for r in grip['rows']:
                if r.get('frame') == wanted: target=r; break
        else:
            row_idx = frame_idx + INDEX_OFFSET
            if 0 <= row_idx < len(grip['rows']): target=grip['rows'][row_idx]
    return {
        'cmd_mm': None if not target else target.get('cmd_mm'),
        'fb_mm': None if not target or not EMIT_FB else target.get('fb_mm'),
        'timestamp_ms': None if not target or not EMIT_TS else target.get('timestamp_ms'),
        '_found': target is not None
    }

# ---------------------------- 整体打乱划分 ----------------------------
def fully_shuffled_split_with_representation(items, tr, vr, te, seed):
    """完全打乱所有数据，但确保测试集包含所有组别的代表性样本"""
    if abs(tr+vr+te-1.0) > 1e-6:
        raise ValueError("train/val/test 比例之和必须为1")
    
    # 首先按组别(object_id+trial_id)分组
    groups = defaultdict(list)
    for item in items:
        group_key = f"{item['object_id']}_{item['trial_id']}"
        groups[group_key].append(item)
    
    # 确保测试集包含每个组别的至少一个样本
    test_items = []
    group_keys = list(groups.keys())
    rnd = random.Random(seed)
    rnd.shuffle(group_keys)
    
    # 从每个组中抽取一个样本到测试集
    for group_key in group_keys:
        if groups[group_key]:
            # 随机选择组内的一个样本
            sample_idx = rnd.randint(0, len(groups[group_key]) - 1)
            test_items.append(groups[group_key].pop(sample_idx))
    
    # 收集剩余的所有样本
    remaining_items = []
    for group_key in group_keys:
        remaining_items.extend(groups[group_key])
    
    # 完全打乱剩余样本
    rnd.shuffle(remaining_items)
    
    # 计算划分点
    total_remaining = len(remaining_items)
    n_train = int(total_remaining * tr / (tr + vr))  # 调整比例，因为测试集已经单独抽取
    n_val = total_remaining - n_train
    
    # 划分训练集和验证集
    train_items = remaining_items[:n_train]
    val_items = remaining_items[n_train:n_train+n_val]
    
    # 合并测试集
    test_items.extend(remaining_items[n_train+n_val:])
    
    # 最后再次打乱测试集以确保随机性
    rnd.shuffle(test_items)
    
    return {
        "train": train_items,
        "val": val_items,
        "test": test_items
    }

# ---------------------------- 主逻辑 ----------------------------
def build_xsense():
    root=Path(XSENSE_ROOT).expanduser().resolve()
    img_exts={e.lower() if e.startswith('.') else f'.{e.lower()}' for e in IMG_EXTS}
    items=[]
    for obj_dir in sorted([d for d in root.iterdir() if d.is_dir()]):
        mass_txt=read_text(obj_dir/"mass.txt") if (obj_dir/"mass.txt").exists() else "NA"
        tex_txt =read_text(obj_dir/"texture.txt") if (obj_dir/"texture.txt").exists() else "NA"
        if NORMALIZE:
            mass_txt=normalize_mass(mass_txt)
            tex_txt=normalize_texture(tex_txt)
        object_id=obj_dir.name
        for trial_dir in sorted([d for d in obj_dir.iterdir() if d.is_dir()]):
            trial_id=trial_dir.name
            grip=read_gripper_log(trial_dir/"gripper_log.csv")
            for side in ("L","R"):
                side_dir=trial_dir/side
                if not side_dir.is_dir(): continue
                start,end,best=parse_three_ints(side_dir/"F.txt")
                frame_map=map_frame_files(side_dir/"difference", img_exts)
                best_fp=frame_map.get(best)
                rng=range(start,end+1) if END_INCLUSIVE else range(start,end)
                for fidx in rng:
                    inp_fp=frame_map.get(fidx)
                    if inp_fp is None: continue
                    extra=get_cmd_by_frame(grip,fidx)
                    rec={
                        "mode":"xsense",
                        "object_id":object_id,
                        "trial_id":trial_id,
                        "side":side,
                        "frame_idx":fidx,
                        "best_idx":best,
                        "mass":mass_txt,
                        "texture":tex_txt,
                        "cmd_mm":extra.get("cmd_mm"),
                        **({"fb_mm":extra.get("fb_mm")} if EMIT_FB else {}),
                        **({"timestamp_ms":extra.get("timestamp_ms")} if EMIT_TS else {}),
                        "input_diff":inp_fp,
                        "contact_depth":side_dir/"contact_depth.png",
                        "target_diff":best_fp,
                        "stem":f"{object_id}_{trial_id}_{side}_{fidx}"
                    }
                    items.append(rec)
    
    # 使用整体打乱划分
    split = fully_shuffled_split_with_representation(items, TRAIN_RATIO, VAL_RATIO, TEST_RATIO, SEED)
    
    outdir=Path(OUT_DIR).resolve(); outdir.mkdir(parents=True,exist_ok=True)
    def mat(p:Path,hint:str)->str: return maybe_materialize(p,root,outdir,PATH_MODE,hint)
    for ss,rows in split.items():
        for r in rows:
            # 为每个文件类型创建基于object_id, trial_id, side的子目录
            input_diff_hint = f"input_diff/{r['object_id']}/{r['trial_id']}/{r['side']}"
            contact_depth_hint = f"contact_depth/{r['object_id']}/{r['trial_id']}/{r['side']}"
            target_diff_hint = f"target_diff/{r['object_id']}/{r['trial_id']}/{r['side']}"

            r["input_diff"] = mat(r["input_diff"], input_diff_hint)
            r["contact_depth"] = mat(r["contact_depth"], contact_depth_hint)
            r["target_diff"] = mat(r["target_diff"], target_diff_hint)
            r["split"] = ss
    return split

# ---------------------------- 输出 ----------------------------
def dump_jsonl(outdir:Path,name:str,rows:List[dict]):
    with (outdir/f"{name}.jsonl").open("w",encoding="utf-8") as f:
        for r in rows: f.write(json.dumps(r,ensure_ascii=False)+"\n")
    print(f"[INFO] 写出 {name}.jsonl: {len(rows)}")

def main():
    split=build_xsense()
    outdir=Path(OUT_DIR).resolve()
    for n in ("train","val","test"): dump_jsonl(outdir,n,split[n])
    
    # 生成详细报告
    report = {
        "num_train": len(split["train"]),
        "num_val": len(split["val"]),
        "num_test": len(split["test"]),
        "total": len(split["train"]) + len(split["val"]) + len(split["test"]),
        "train_ratio": len(split["train"]) / (len(split["train"]) + len(split["val"]) + len(split["test"])),
        "val_ratio": len(split["val"]) / (len(split["train"]) + len(split["val"]) + len(split["test"])),
        "test_ratio": len(split["test"]) / (len(split["train"]) + len(split["val"]) + len(split["test"])),
        "seed": SEED,
        "split_method": "fully_shuffled_with_group_representation",
        "path_mode": PATH_MODE
    }
    
    (outdir/"report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("[OK] 完成，输出目录：", outdir)
    print(f"[INFO] 训练集: {len(split['train'])} 样本")
    print(f"[INFO] 验证集: {len(split['val'])} 样本")
    print(f"[INFO] 测试集: {len(split['test'])} 样本")
    print(f"[INFO] 路径模式: {PATH_MODE}")

if __name__=="__main__":
    main()