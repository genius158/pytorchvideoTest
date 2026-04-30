"""
将 SoccerNet BAS-2025 标注事件离线导出为真实 mp4 小片段。

数据集结构示例:
  <data_root>/
	train/.../<game>/224p.mp4
	train/.../<game>/Labels-ball.json
	valid/.../<game>/...

用法:
  python getvideo.py \
	--data_root /path/to/SN-BAS-2025 \
	--output_root /path/to/SN-BAS-2025-clips \
	--clip_duration 1.6

可选:
  --splits train valid
  --resize 224
  --overwrite
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import av


SOCCERNET_ACTIONS = [
	"PASS",
	"DRIVE",
	"HIGH PASS",
	"HEADER",
	"SHOT",
	"CROSS",
	"THROW IN",
	"FREE KICK",
	"GOAL",
	"OUT",
	"PLAYER SUCCESSFUL TACKLE",
	"BALL PLAYER BLOCK",
]
ACTION_TO_IDX = {action: idx for idx, action in enumerate(SOCCERNET_ACTIONS)}


def parse_args() -> argparse.Namespace:
	p = argparse.ArgumentParser(description="导出 SoccerNet 事件为 mp4 片段")
	p.add_argument("--data_root", type=str, required=True, help="根目录（包含 train/valid）")
	p.add_argument("--output_root", type=str, required=True, help="导出目录")
	p.add_argument(
		"--splits",
		nargs="+",
		default=["train", "valid"],
		help="需要导出的 split 列表，默认: train valid",
	)
	p.add_argument("--clip_duration", type=float, default=1.6, help="片段时长（秒），默认 1.6")
	p.add_argument(
		"--codec",
		type=str,
		default="mpeg4",
		help="视频编码器，默认 mpeg4（兼容性高）；可尝试 libx264",
	)
	p.add_argument("--fps", type=float, default=0.0, help="强制导出 fps，0 表示跟随源视频")
	p.add_argument(
		"--resize",
		type=int,
		default=0,
		help="输出分辨率边长（正方形），0 表示不缩放，默认 0",
	)
	p.add_argument("--overwrite", action="store_true", help="覆盖已存在的小片段")
	p.add_argument(
		"--allow_unknown_labels",
		action="store_true",
		help="导出未知标签（label_idx=-1）；默认仅导出 12 个定义动作",
	)
	return p.parse_args()


def sanitize_name(text: str) -> str:
	text = text.strip().replace(" ", "_")
	text = re.sub(r"[^\w\-.]+", "_", text)
	return text[:120] if text else "unknown"


def find_video_file(game_dir: Path) -> Optional[Path]:
	p224 = game_dir / "224p.mp4"
	p720 = game_dir / "720p.mp4"
	if p224.exists():
		return p224
	if p720.exists():
		return p720
	# 兜底：找首个 mp4
	for f in sorted(game_dir.glob("*.mp4")):
		return f
	return None


def iter_events(label_path: Path, allow_unknown_labels: bool) -> Iterable[Tuple[float, str, int, int]]:
	with label_path.open("r", encoding="utf-8") as f:
		data = json.load(f)

	for ann in data.get("annotations", []):
		label = str(ann.get("label", "")).strip()
		try:
			position_ms = int(ann["position"])
		except (KeyError, ValueError, TypeError):
			continue

		if label in ACTION_TO_IDX:
			label_idx = ACTION_TO_IDX[label]
		elif allow_unknown_labels:
			label_idx = -1
		else:
			continue

		yield position_ms / 1000.0, label, label_idx, position_ms


def _safe_fps(stream: av.video.stream.VideoStream, fallback: float = 25.0) -> float:
	if stream.average_rate is not None:
		try:
			fps = float(stream.average_rate)
			if fps > 0:
				return fps
		except Exception:
			pass
	if stream.base_rate is not None:
		try:
			fps = float(stream.base_rate)
			if fps > 0:
				return fps
		except Exception:
			pass
	return fallback


def export_clip(
	video_path: Path,
	out_path: Path,
	start_sec: float,
	end_sec: float,
	codec: str,
	force_fps: float,
	resize: int,
) -> int:
	"""导出 [start_sec, end_sec] 片段，返回导出帧数。"""
	out_path.parent.mkdir(parents=True, exist_ok=True)

	in_container = av.open(str(video_path))
	in_stream = in_container.streams.video[0]

	fps = force_fps if force_fps > 0 else _safe_fps(in_stream)

	# seek 到起始附近，提高解码效率
	seek_ts = int(max(start_sec, 0.0) / in_stream.time_base)
	in_container.seek(seek_ts, stream=in_stream)

	collected = []
	for frame in in_container.decode(video=0):
		if frame.pts is None:
			continue
		t = float(frame.pts * in_stream.time_base)
		if t < start_sec:
			continue
		if t > end_sec:
			break
		collected.append(frame)

	in_container.close()

	if not collected:
		return 0

	first = collected[0]
	if resize > 0:
		width = resize
		height = resize
	else:
		width = first.width
		height = first.height

	out_container = av.open(str(out_path), mode="w")
	out_stream = out_container.add_stream(codec, rate=fps)
	out_stream.width = width
	out_stream.height = height
	out_stream.pix_fmt = "yuv420p"

	written = 0
	for frame in collected:
		if resize > 0:
			frame = frame.reformat(width=width, height=height)
		if frame.format.name != "yuv420p":
			frame = frame.reformat(format="yuv420p")
		for packet in out_stream.encode(frame):
			out_container.mux(packet)
		written += 1

	for packet in out_stream.encode(None):
		out_container.mux(packet)
	out_container.close()
	return written


def process_split(
	data_root: Path,
	output_root: Path,
	split: str,
	clip_duration: float,
	codec: str,
	fps: float,
	resize: int,
	overwrite: bool,
	allow_unknown_labels: bool,
) -> List[dict]:
	split_root = data_root / split
	if not split_root.exists():
		print(f"[跳过] split 不存在: {split_root}")
		return []

	rows: List[dict] = []
	label_files = sorted(split_root.rglob("Labels-ball.json"))
	print(f"[{split}] 找到标注文件 {len(label_files)} 个")

	for label_path in label_files:
		game_dir = label_path.parent
		video_path = find_video_file(game_dir)
		if video_path is None:
			print(f"[警告] 缺少 mp4: {game_dir}")
			continue

		rel_game = game_dir.relative_to(split_root)
		out_game_dir = output_root / split / rel_game

		event_id = 0
		for center_sec, label, label_idx, position_ms in iter_events(label_path, allow_unknown_labels):
			event_id += 1
			half = clip_duration / 2.0
			start_sec = max(0.0, center_sec - half)
			end_sec = center_sec + half

			label_name = sanitize_name(label)
			out_name = f"{event_id:06d}_{label_name}_{position_ms}ms.mp4"
			out_path = out_game_dir / out_name

			if out_path.exists() and not overwrite:
				frame_count = -1
				status = "exists"
			else:
				try:
					frame_count = export_clip(
						video_path=video_path,
						out_path=out_path,
						start_sec=start_sec,
						end_sec=end_sec,
						codec=codec,
						force_fps=fps,
						resize=resize,
					)
					status = "ok" if frame_count > 0 else "empty"
				except Exception as e:
					frame_count = 0
					status = f"error: {e}"

			rows.append(
				{
					"split": split,
					"source_video": str(video_path),
					"label": label,
					"label_idx": label_idx,
					"position_ms": position_ms,
					"center_sec": f"{center_sec:.3f}",
					"start_sec": f"{start_sec:.3f}",
					"end_sec": f"{end_sec:.3f}",
					"clip_path": str(out_path),
					"frame_count": frame_count,
					"status": status,
				}
			)

	return rows


def write_manifest(rows: List[dict], output_root: Path) -> None:
	manifest_path = output_root / "manifest.csv"
	output_root.mkdir(parents=True, exist_ok=True)

	fieldnames = [
		"split",
		"source_video",
		"label",
		"label_idx",
		"position_ms",
		"center_sec",
		"start_sec",
		"end_sec",
		"clip_path",
		"frame_count",
		"status",
	]
	with manifest_path.open("w", newline="", encoding="utf-8") as f:
		writer = csv.DictWriter(f, fieldnames=fieldnames)
		writer.writeheader()
		writer.writerows(rows)

	ok = sum(1 for r in rows if r["status"] == "ok")
	exists = sum(1 for r in rows if r["status"] == "exists")
	empty = sum(1 for r in rows if r["status"] == "empty")
	err = len(rows) - ok - exists - empty
	print("\n=== 导出完成 ===")
	print(f"总事件数: {len(rows)}")
	print(f"成功导出: {ok}")
	print(f"已存在跳过: {exists}")
	print(f"空片段: {empty}")
	print(f"错误: {err}")
	print(f"清单文件: {manifest_path}")


def main() -> None:
	args = parse_args()

	data_root = Path(args.data_root)
	output_root = Path(args.output_root)

	all_rows: List[dict] = []
	for split in args.splits:
		rows = process_split(
			data_root=data_root,
			output_root=output_root,
			split=split,
			clip_duration=float(args.clip_duration),
			codec=args.codec,
			fps=float(args.fps),
			resize=int(args.resize),
			overwrite=bool(args.overwrite),
			allow_unknown_labels=bool(args.allow_unknown_labels),
		)
		all_rows.extend(rows)

	write_manifest(all_rows, output_root)


if __name__ == "__main__":
	main()

