import os
from argparse import ArgumentParser
from pathlib import Path
import shutil


def copy_frames(input_dir: Path, output_dir: Path):
    # Create the output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)

    # Copy all files from input_dir to output_dir
    for file in input_dir.glob("*"):
        print(f"Copying {file} to {output_dir}")
        tmp_path = file/"Frames"
        for image in sorted(tmp_path.glob("*.jpg")):
            # image_path = os.path.join(tmp_path, file)
            shutil.copy(image, os.path.join(output_dir, image.name))


if __name__ == "__main__":
    parser = ArgumentParser(description="Gather all jpgs from dmt recordings.")
    parser.add_argument(
        "--dmt_path",
        type=Path,
        required=True,
        help="Path to DMT File",
    )
    args = parser.parse_args()

    input_dir = Path(os.path.join(args.dmt_path, "datasets"))
    output_dir = Path(os.path.join(args.dmt_path, "Frames"))
    copy_frames(input_dir, output_dir)