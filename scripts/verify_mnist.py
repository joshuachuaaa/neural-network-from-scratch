from __future__ import annotations

import hashlib
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MANIFEST = PROJECT_ROOT / "data" / "mnist" / "SHA256SUMS"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    failures = []
    for line in MANIFEST.read_text().splitlines():
        if not line.strip():
            continue
        expected, relative_path = line.split(maxsplit=1)
        path = PROJECT_ROOT / relative_path
        actual = sha256(path)
        if actual != expected:
            failures.append(f"{relative_path}: expected {expected}, got {actual}")

    if failures:
        raise SystemExit("\n".join(failures))

    print("MNIST checksums verified.")


if __name__ == "__main__":
    main()

