from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from neural_network_from_scratch.train import main


if __name__ == "__main__":
    main()
