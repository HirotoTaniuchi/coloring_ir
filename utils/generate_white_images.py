#!/usr/bin/env python3
"""
入力ディレクトリ配下の画像ごとに、同サイズ・同ファイル名の真っ白なRGB画像(255,255,255)を
出力ディレクトリ配下に生成するCLIツール。

特徴:
- ディレクトリ再帰探索 (--recursive)
- 対象拡張子の指定 (--exts)
- 上書き制御 (--overwrite)
- 事前確認のためのドライラン (--dry-run)
- 処理件数制限 (--limit)

使い方例:
  python projects/coloring_ir/utils/generate_white_images.py \
    /path/to/src /path/to/dst --recursive --dry-run --limit 5

備考:
- Pillow(PIL) は実際に画像を書き出す時にのみ必要です。--dry-run では未使用です。
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple


DEFAULT_EXTS = [
    ".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp",
]


def parse_exts(exts_arg: Sequence[str] | None) -> List[str]:
    """正規化された拡張子リスト（ドット付き・小文字）を返す。

    --exts は空白区切り or カンマ区切りの両方を許可する。
    未指定なら既定の拡張子を返す。
    """
    if not exts_arg:
        return DEFAULT_EXTS.copy()

    raw: List[str] = []
    for item in exts_arg:
        raw.extend([p for p in item.split(",") if p])

    norm = []
    for e in raw:
        e = e.strip().lower()
        if not e:
            continue
        if not e.startswith('.'):
            e = '.' + e
        norm.append(e)
    return norm or DEFAULT_EXTS.copy()


def iter_image_files(src_dir: Path, recursive: bool, exts: Sequence[str]) -> Iterable[Path]:
    """src_dir配下の対象拡張子ファイルを列挙する。"""
    if recursive:
        for p in src_dir.rglob('*'):
            if p.is_file() and p.suffix.lower() in exts:
                yield p
    else:
        for p in src_dir.iterdir():
            if p.is_file() and p.suffix.lower() in exts:
                yield p


def ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def make_white_image_file(size: Tuple[int, int], out_path: Path) -> None:
    """指定サイズの真っ白RGB画像を out_path に保存する。

    Pillowのインポートはこの関数内に限定し、--dry-run 時にはインポートを避ける。
    """
    from PIL import Image  # 局所インポート

    width, height = size
    img = Image.new('RGB', (width, height), (255, 255, 255))
    # JPEGの画質などはデフォルト。必要なら拡張子に応じてオプション調整も可能。
    img.save(out_path)


def get_image_size(image_path: Path) -> Tuple[int, int]:
    """画像サイズを (width, height) で返す。"""
    from PIL import Image  # 局所インポート

    with Image.open(image_path) as im:
        return im.size  # (width, height)


def process(
    src_dir: Path,
    dst_dir: Path,
    recursive: bool,
    exts: Sequence[str],
    overwrite: bool,
    dry_run: bool,
    limit: int | None,
) -> int:
    """主処理。生成した(または予定の)ファイル数を返す。"""
    count = 0
    for in_path in iter_image_files(src_dir, recursive, exts):
        rel = in_path.relative_to(src_dir)
        out_path = dst_dir / rel

        if not overwrite and out_path.exists():
            print(f"skip (exists): {out_path}")
            continue

        if dry_run:
            print(f"DRY-RUN: {in_path} -> {out_path}")
        else:
            # サイズ取得し、白画像を書き出す
            size = get_image_size(in_path)
            ensure_parent_dir(out_path)
            make_white_image_file(size, out_path)
            print(f"wrote: {out_path}")

        count += 1
        if limit is not None and limit > 0 and count >= limit:
            break

    return count


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "入力ディレクトリ配下の画像に対応する真っ白RGB画像を、同サイズ・同名で出力します。"
        )
    )
    p.add_argument(
        "src_dir", type=Path,
        help="入力ディレクトリ (元画像が格納されているルート)"
    )
    p.add_argument(
        "dst_dir", type=Path,
        help="出力ディレクトリ (白画像を書き出すルート)"
    )
    p.add_argument(
        "--recursive", action="store_true",
        help="サブディレクトリも再帰的に処理する"
    )
    p.add_argument(
        "--exts", nargs="*",
        help=(
            "対象拡張子（空白またはカンマ区切り）。例: --exts jpg png,webp\n"
            f"未指定時の既定値: {', '.join(DEFAULT_EXTS)}"
        )
    )
    p.add_argument(
        "--overwrite", action="store_true",
        help="出力先に同名ファイルがある場合に上書きする"
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="実際には書き込まず、予定だけ表示する"
    )
    p.add_argument(
        "--limit", type=int, default=None,
        help="最大処理件数を制限する (デバッグ用途)"
    )
    return p


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    src_dir: Path = args.src_dir
    dst_dir: Path = args.dst_dir

    if not src_dir.exists() or not src_dir.is_dir():
        parser.error(f"src_dir が存在しないかディレクトリではありません: {src_dir}")

    exts = parse_exts(args.exts)

    count = process(
        src_dir=src_dir,
        dst_dir=dst_dir,
        recursive=bool(args.recursive),
        exts=exts,
        overwrite=bool(args.overwrite),
        dry_run=bool(args.dry_run),
        limit=args.limit,
    )

    if args.dry_run:
        print(f"DRY-RUN summary: {count} files would be written.")
    else:
        print(f"Done: {count} files written.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
