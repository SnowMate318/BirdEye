from __future__ import annotations

from pathlib import Path
import re


_INDEX_RE = re.compile(r"(\d+)_img\.(?:png|jpg|jpeg)$", re.IGNORECASE)


def read_nyu_split(path: Path) -> list[int]:
    """DSINE에 포함된 NYU split txt에서 MAT index를 읽는다.

    split file 행 예시는 `train/000002_img.png`이며, 숫자 2가
    `nyu_depth_v2_labeled.mat`의 0-based frame index로 쓰인다.
    """

    if not path.exists():
        raise FileNotFoundError(f"NYU split file not found: {path}")
    indices: list[int] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        item = line.strip()
        if not item:
            continue
        match = _INDEX_RE.search(item)
        if match is None:
            raise ValueError(f"Cannot parse NYU split line: {item}")
        indices.append(int(match.group(1)))
    return indices
