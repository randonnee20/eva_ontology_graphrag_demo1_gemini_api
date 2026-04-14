"""utils/config.py — 중앙 설정 관리자"""

from pathlib import Path
import yaml

_config = None
_root = None


def get_root() -> Path:
    global _root
    if _root is None:
        current = Path(__file__).resolve().parent
        for _ in range(6):
            if (current / "config.yaml").exists():
                _root = current
                return _root
            current = current.parent
        raise FileNotFoundError("config.yaml을 찾을 수 없음")
    return _root


def get_config() -> dict:
    global _config
    if _config is None:
        root = get_root()
        with open(root / "config.yaml", encoding="utf-8") as f:
            _config = yaml.safe_load(f)

        # 경로를 절대 경로로 변환 + 디렉터리 생성
        for key, rel in _config.get("paths", {}).items():
            if key == "ontology":
                abs_path = root / rel
                abs_path.parent.mkdir(parents=True, exist_ok=True)
            else:
                abs_path = root / rel
                abs_path.mkdir(parents=True, exist_ok=True)
            _config["paths"][key] = str(abs_path)

    return _config
