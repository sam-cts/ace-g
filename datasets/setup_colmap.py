import os
from argparse import ArgumentParser
from pathlib import Path
import numpy as np
import pycolmap
import shutil
import math


def save_calibration(reconstruction: pycolmap.Reconstruction, out_file: Path):
    # We use the first (and only) camera.

    camera_idx = 1
    # We save the average of the focal length.
    calibration = reconstruction.cameras[camera_idx].calibration_matrix()
    focal_length = (calibration[0, 0] + calibration[1, 1]) / 2

    with out_file.open("w") as f:
        f.write(f"{focal_length}\n")
    print(f"Saved focal length ({focal_length:.3f}) to: {out_file}")


def process_colmap_scene(in_dir: Path, colmap_dir: Path, out_dir: Path):
    # We work in a temporary folder then rename at the end.
    temp_out_path = out_dir.with_suffix(".tmp")
    temp_out_path.mkdir(exist_ok=True)

    # Load reconstruction
    reconstruction = pycolmap.Reconstruction(colmap_dir)
    print(
        f"Loaded reconstruction from: {colmap_dir} with"
        f" {len(reconstruction.cameras)} cameras and {len(reconstruction.images)} images."
    )

    # Not all images have a pose, so we create a map: image_name -> image_id to check if a pose exists.
    image_name_to_id = {Path(v.name).name: k for k, v in reconstruction.images.items()}

    # Export focal length.
    out_calibration_path = temp_out_path / "focal_length.txt"
    save_calibration(reconstruction, out_calibration_path)

    # Process each image.
    for image_path in sorted(in_dir.glob("*.jpg")):
        image_name = image_path.name

        out_image_file = temp_out_path / image_name
        out_pose_file = temp_out_path / image_name.replace(".jpg", "_pose.txt")

        if image_name in image_name_to_id:
            # Extract metadata from reconstruction.
            image_id = image_name_to_id[image_name]
            image = reconstruction.images[image_id]

            # Colmap stores world to camera'.
            cam_from_world = np.eye(4)
            cam_from_world[:3, :4] = image.cam_from_world().matrix()
            # # ACE uses camera to world.
            camera_to_world = np.linalg.inv(cam_from_world)

        else:
            # Image wasn't reconstructed, create dummy pose matrix.
            camera_to_world = np.full((4, 4), np.inf)

        # Save pose.
        with out_pose_file.open("w") as f:
            np.savetxt(f, camera_to_world)

        # Create symlink.
        if out_image_file.exists():
            out_image_file.unlink()
        out_image_file.symlink_to(os.path.relpath(image_path, start=temp_out_path))

    # If we got here there were no errors. Rename to the final folder.
    print(f"Renaming {temp_out_path} to {out_dir}")
    temp_out_path.rename(out_dir)


def separate_files(source_folder, destination_folder):
    # Define target directories within the destination folder
    rgb_folder = os.path.join(destination_folder, "rgb")
    poses_folder = os.path.join(destination_folder, "poses")
    calibration_folder = os.path.join(destination_folder, "calibration")

    # Create destination directories if they don't exist
    os.makedirs(rgb_folder, exist_ok=True)
    os.makedirs(poses_folder, exist_ok=True)
    os.makedirs(calibration_folder, exist_ok=True)

    # Iterate over files in the source folder
    for file in os.listdir(source_folder):
        file_path = os.path.join(source_folder, file)

        # Copy .jpg files to 'rgb' folder
        if file.lower().endswith(".jpg"):
            shutil.copy(file_path, os.path.join(rgb_folder, file))

        # Copy .txt files to 'poses' folder
        elif file.lower().endswith(".txt"):
            shutil.copy(file_path, os.path.join(poses_folder, file))

    # Move "focal_length.txt" from poses folder to calibration folder
    focal_length_path = os.path.join(poses_folder, "focal_length.txt")
    if os.path.exists(focal_length_path):
        shutil.move(focal_length_path, os.path.join(calibration_folder, "focal_length.txt"))

    print("Files successfully copied to:", destination_folder)
    

def copy_and_cleanup(source_folder):
    # Define destination directories
    train_folder = os.path.join(source_folder, "train")
    test_folder = os.path.join(source_folder, "test")

    # Create train and test directories if they don't exist
    os.makedirs(train_folder, exist_ok=True)
    os.makedirs(test_folder, exist_ok=True)

    # List of folders to copy
    folders_to_copy = ["calibration", "poses", "rgb"]

    for folder in folders_to_copy:
        source_path = os.path.join(source_folder, folder)
        
        if os.path.exists(source_path):
            # Copy the entire folder to both "train" and "test" directories
            shutil.copytree(source_path, os.path.join(train_folder, folder))
            shutil.copytree(source_path, os.path.join(test_folder, folder))

            # Remove the original folder after copying
            shutil.rmtree(source_path)

    print("Files successfully copied to 'train' and 'test' folders, and original folders removed!")


def copy_and_cleanup_split(source_folder, train_ratio=0.8
):
    # Define destination directories
    train_folder = os.path.join(source_folder, "train")
    test_folder = os.path.join(source_folder, "test")

    # Create train and test directories if they don't exist
    os.makedirs(train_folder, exist_ok=True)
    os.makedirs(test_folder, exist_ok=True)

    shutil.copytree(
        os.path.join(source_folder, "calibration"), 
        os.path.join(train_folder, "calibration"))
    shutil.copytree(
        os.path.join(source_folder, "calibration"), 
        os.path.join(test_folder, "calibration"))
    shutil.rmtree(os.path.join(source_folder, "calibration"))

    image_folder = os.path.join(source_folder, "rgb")
    image_files = sorted([f for f in os.listdir(image_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    total_images = len(image_files)

    pose_folder = os.path.join(source_folder, "poses")
    pose_files = sorted([f for f in os.listdir(pose_folder) if f.lower().endswith(('.txt'))])
    total_poses = len(pose_files)

    if total_images != total_poses:
        print(f"miss match numbers of images and poses, abort copying!!! total images: {total_images} total poses: {total_poses}")

    num_to_copy = math.ceil(total_images * train_ratio)
    # Compute step size for uniform sampling
    step = total_images / num_to_copy
    selected_indices = [int(i * step) for i in range(num_to_copy)]
    
    train_rgb_folder = os.path.join(train_folder, "rgb")
    train_poses_folder = os.path.join(train_folder, "poses")
    test_rgb_folder = os.path.join(test_folder, "rgb")
    test_poses_folder = os.path.join(test_folder, "poses")
    
    os.makedirs(train_rgb_folder, exist_ok=True)
    os.makedirs(train_poses_folder, exist_ok=True)
    os.makedirs(test_rgb_folder, exist_ok=True)
    os.makedirs(test_poses_folder, exist_ok=True)

    for i in range(total_images):
        if i in selected_indices:
            shutil.copy2(os.path.join(image_folder, image_files[i]), os.path.join(train_rgb_folder, image_files[i]))
            shutil.copy2(os.path.join(pose_folder, pose_files[i]), os.path.join(train_poses_folder, pose_files[i]))
        else:
            shutil.copy2(os.path.join(image_folder, image_files[i]), os.path.join(test_rgb_folder, image_files[i]))
            shutil.copy2(os.path.join(pose_folder, pose_files[i]), os.path.join(test_poses_folder, pose_files[i]))


    shutil.rmtree(os.path.join(source_folder, "rgb"))
    shutil.rmtree(os.path.join(source_folder, "poses"))
    print("Files successfully copied to 'train' and 'test' folders, and original folders removed!")


if __name__ == "__main__":
    parser = ArgumentParser(description="Setup the Tanks and Temples dataset.")
    parser.add_argument(
        "--colmap_path",
        type=Path,
        help="Path to COLMAP-format files",
    )
    parser.add_argument(
        "--raw_image_path",
        type=Path,
        help="Path to raw images",
    )
    parser.add_argument(
        "--output_path",
        type=Path,
        help="Path to output files",
    )
    parser.add_argument(
        '--global_setup', 
        type=bool, 
        default=False,
        help='create a video of the mapping process'
    )
    parser.add_argument(
        '--train_set_ratio', 
        type=float, 
        default=0.8,
        help='ratio of training set, default 0.8'
    )

    args = parser.parse_args()

    if os.path.isdir(args.output_path):
        print(f"output path exists '{args.output_path}', removing folder")
        try:
            shutil.rmtree(args.output_path)
        except OSError as e:
            print(f"Error removing folder '{args.output_path}': {e}")

    process_colmap_scene(args.raw_image_path, args.colmap_path, args.output_path)

    if args.global_setup:
        print("Separating files into 'rgb' and 'poses' folders...")
        separate_files(args.output_path, args.output_path.parent / "global_setup")
        copy_and_cleanup_split(args.output_path.parent / "global_setup", args.train_set_ratio)