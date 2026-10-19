import argparse
import json
from .config import load_config
from .data.frames import frame_paths


def main():
    parser = argparse.ArgumentParser(prog="ultralabel")
    commands = parser.add_subparsers(dest="command", required=True)
    validate = commands.add_parser("validate-config")
    validate.add_argument("--config", default="configs/default.yaml")
    inspect = commands.add_parser("inspect-frames")
    inspect.add_argument("directory")
    args = parser.parse_args()
    if args.command == "validate-config":
        print(json.dumps(load_config(args.config).to_dict(), indent=2))
    else:
        from PIL import Image
        paths = frame_paths(args.directory)
        with Image.open(paths[0]) as image:
            print(json.dumps({"frames": len(paths), "first_frame": str(paths[0]),
                              "width": image.width, "height": image.height}, indent=2))


if __name__ == "__main__":
    main()
