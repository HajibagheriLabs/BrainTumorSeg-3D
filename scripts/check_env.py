"""Print the library versions and the GPU that training will run on."""

import platform

import monai
import torch


def _row(label: str, value: object) -> None:
    print(f"{label:<16}{value}")


def main() -> None:
    _row("python", platform.python_version())
    _row("torch", f"{torch.__version__} (cuda build {torch.version.cuda})")
    _row("monai", monai.__version__)
    _row("cuda available", torch.cuda.is_available())
    if not torch.cuda.is_available():
        print("no cuda device: training needs one, tests and evaluation can run on cpu")
        return
    _row("cudnn", torch.backends.cudnn.version())
    for index in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(index)
        vram = f"{props.total_memory / 1024**3:.1f} GiB"
        _row(f"gpu {index}", f"{props.name}, {vram}, sm_{props.major}{props.minor}")


if __name__ == "__main__":
    main()
