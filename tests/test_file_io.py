from pathlib import Path

import numpy as np

from src.utils.file_io import ensure_directory
from src.utils.file_io import load_json
from src.utils.file_io import save_json


def test_save_json_handles_numpy(tmp_path: Path):
    out_dir = tmp_path / "artifacts"
    ensure_directory(out_dir)
    out_file = out_dir / "stats.json"

    data = {
        "count": np.int64(5),
        "ratio": np.float64(0.25),
        "ok": np.bool_(True),
        "arr": np.array([1, 2, 3], dtype=np.int64),
    }

    save_json(data, out_file)
    loaded = load_json(out_file)

    assert loaded["count"] == 5
    assert abs(loaded["ratio"] - 0.25) < 1e-9
    assert loaded["ok"] is True
    assert loaded["arr"] == [1, 2, 3]
