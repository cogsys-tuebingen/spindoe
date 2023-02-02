from pathlib import Path
import argparse
import re
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d",
        "--dir",
        default=Path(
            "../data/test/", help="Directory where the sequential ball images are saved"
        ),
    )
    args = parser.parse_args()
    dir_path = Path(args.dir)
    img_paths = dir_path.glob("*.png")
    for path in img_paths:
        # Get the timestamp of the image
        result = re.search("time(.*)timestamp", str(path))
        t = float(result.group(1))
        t /= 1000
        # Generate the new name of the image
        time_parts = str(t).split(".")
        seconds = time_parts[0]
        microseconds = time_parts[1][:9]
        name = "s".join([seconds, microseconds]) + ".png"
        # print(name)
        # Rename the image
        os.rename(str(path.absolute()), str(path.parent.absolute() / name))
