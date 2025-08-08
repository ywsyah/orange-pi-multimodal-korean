
# 1. 读入你的 converted_mllm_data.json
input_path = r"E:\Desktop\ai相关资料\学习资料路线等\中科院软件实习项目\项目工程文件测试\数据集\OpenDataLab___WanJuanSiLu2O\raw\image\ko\ko_image_caption.jsonl"
output_path = r"E:\Desktop\ai相关资料\学习资料路线等\中科院软件实习项目\项目工程文件测试\数据集\OpenDataLab___WanJuanSiLu2O\raw\image\ko\converted_mllm_data_local_ko_all.json"

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从形如下列结构的 JSONL 文件逐行读取：
{
  "img_id": "0005861388c62ae3...",
  "image": {
    "path": "https://...jpg",
    "resolution": [1280, 720],
    "size": 166.0166015625,
    "format": "JPEG"
  },
  "captions": {...},
  "labels": {...}
}

将图片下载到本地目录，若下载成功：
- 仅将 `image.path` 替换为本地文件路径（其余键值保持不变）。
- 将该行写入新的 JSONL。
若下载失败：
- 跳过该行，不写入输出文件。

支持多线程下载、超时与重试，并自动创建缺失目录。
"""

import argparse
import concurrent.futures as futures
import hashlib
import json
import os
from pathlib import Path
from typing import Optional, Tuple
from urllib.parse import urlparse

import requests
from requests.adapters import HTTPAdapter, Retry
from tqdm import tqdm


FORMAT_EXT_MAP = {
    "JPEG": ".jpg",
    "JPG": ".jpg",
    "PNG": ".png",
    "WEBP": ".webp",
    "BMP": ".bmp",
    "GIF": ".gif",
}


def guess_ext(url: str, fmt: Optional[str]) -> str:
    """根据 URL 后缀或 image.format 猜测扩展名，默认 .jpg。"""
    path = urlparse(url).path
    suf = Path(path).suffix.lower()
    if suf in {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"}:
        return ".jpg" if suf == ".jpeg" else suf
    if fmt:
        ext = FORMAT_EXT_MAP.get(fmt.upper())
        if ext:
            return ext
    return ".jpg"


def build_local_path(out_dir: Path, img_id: Optional[str], url: str, fmt: Optional[str]) -> Path:
    """构造稳定且不冲突的本地文件路径。
    优先使用 img_id；若无则用 URL 的 SHA1。
    为避免单目录过多文件，按前两位做子目录分桶。
    """
    ext = guess_ext(url, fmt)
    if img_id and isinstance(img_id, str) and len(img_id) >= 2:
        key = img_id
    else:
        key = hashlib.sha1(url.encode("utf-8")).hexdigest()
    subdir = key[:2]
    return out_dir / subdir / f"{key}{ext}"


def make_session() -> requests.Session:
    sess = requests.Session()
    retries = Retry(
        total=3,
        backoff_factor=0.5,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["HEAD", "GET", "OPTIONS"]
    )
    adapter = HTTPAdapter(max_retries=retries, pool_connections=64, pool_maxsize=64)
    sess.mount("http://", adapter)
    sess.mount("https://", adapter)
    sess.headers.update({
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Python-Requests Downloader"
    })
    return sess


def download_one(
    line: str,
    out_dir: Path,
    keep_remote: bool,
    timeout: float,
    verify_ssl: bool,
) -> Tuple[bool, Optional[str]]:
    """处理一行 JSON：下载成功则返回 (True, 输出的 JSON 行字符串)；失败返回 (False, None)。"""
    try:
        rec = json.loads(line)
    except Exception:
        return False, None

    image = rec.get("image", {}) or {}
    url = image.get("path")
    if not isinstance(url, str) or not url.startswith("http"):
        return False, None

    img_id = rec.get("img_id")
    fmt = image.get("format")
    local_path = build_local_path(out_dir, img_id, url, fmt)
    local_path.parent.mkdir(parents=True, exist_ok=True)

    session = make_session()
    try:
        with session.get(url, stream=True, timeout=timeout, verify=verify_ssl) as resp:
            resp.raise_for_status()
            tmp_path = local_path.with_suffix(local_path.suffix + ".part")
            with open(tmp_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)
            os.replace(tmp_path, local_path)
    except Exception:
        # 下载失败，清理临时文件
        try:
            tmp_path = local_path.with_suffix(local_path.suffix + ".part")
            if tmp_path.exists():
                tmp_path.unlink(missing_ok=True)  # type: ignore[arg-type]
        except Exception:
            pass
        return False, None

    # 写出记录：仅替换 image.path，其余字段保持不变
    image["path"] = str(local_path.as_posix())
    if not keep_remote:
        # 不新增任何字段
        pass
    else:
        # 可选：保留原始 URL
        image.setdefault("remote_path", url)
    rec["image"] = image

    try:
        return True, json.dumps(rec, ensure_ascii=False)
    except Exception:
        return False, None


def main():
    parser = argparse.ArgumentParser(description="Download images from JSONL and rewrite local paths")
    parser.add_argument("--input", default=input_path, help="输入 JSONL 文件路径")
    parser.add_argument("--output", default=output_path, help="输出 JSONL 文件路径（仅成功项）")
    parser.add_argument("--out_dir", default="data_ko_all/images", help="图片保存根目录")
    parser.add_argument("--workers", type=int, default=64, help="下载并发线程数")
    parser.add_argument("--timeout", type=float, default=1.0, help="单请求超时（秒）")
    parser.add_argument("--keep-remote", action="store_true", help="在 image.remote_path 中保留原始 URL")
    parser.add_argument("--no-verify-ssl", action="store_true", help="禁用 SSL 证书验证（不建议，某些自签名站点可用）")
    parser.add_argument("--max-lines", type=int, default=0, help="仅处理前 N 行（0 表示处理全部）")
    args = parser.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 逐行读取，支持仅处理前 N 行，避免一次性载入超大文件
    lines = []
    with in_path.open("r", encoding="utf-8") as fr:
        if args.max_lines and args.max_lines > 0:
            from itertools import islice
            for line in islice(fr, args.max_lines):
                lines.append(line.rstrip(""))
        else:
            for line in fr:
                lines.append(line.rstrip(""))

    success = 0
    failed = 0

    with out_path.open("w", encoding="utf-8") as fw:
        with futures.ThreadPoolExecutor(max_workers=args.workers) as ex:
            tasks = [
                ex.submit(
                    download_one,
                    line,
                    out_dir,
                    args.keep_remote,
                    args.timeout,
                    not args.no_verify_ssl,
                ) for line in lines
            ]
            for fut in tqdm(futures.as_completed(tasks), total=len(tasks), desc="Downloading"):
                ok, out_line = fut.result()
                if ok and out_line:
                    fw.write(out_line + "\n")
                    success += 1
                else:
                    failed += 1

    print(f"Done. success={success}, failed={failed}, output={out_path}")


if __name__ == "__main__":
    main()