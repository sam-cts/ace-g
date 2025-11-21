import sys
import cv2
import numpy as np
import math
import tqdm
import yaml
from pathlib import Path

from domain.domain import Domain
from ace_g import data_io, eval_poses_utils

T_domain_to_ROS = np.array([
    [1, 0, 0, 0],
    [0, 0, -1, 0],
    [0, 1, 0, 0],
    [0, 0, 0, 1]
])

def get_domain_map(config_file):

    with open(config_file, 'r') as file:
        domain_config = yaml.safe_load(file)

    auki_domain = Domain(domain_config)
    ret, msg = auki_domain.auth()
    if not ret:
        print(f"domain authentication failed. message: {msg}")
        sys.exit(1)
    map_resolution = 50
    image, metadata = auki_domain.get_map(resolution=map_resolution)
    if image is None:
        print(f"no image returned")
        sys.exit(1)

    cv_image = np.asarray(image)
    return cv_image.copy(), metadata


def map_value_to_color(value, vmin=0.0, vmax=2.0):
    """Map a float value to an RGB color using OpenCV's JET colormap."""
    norm = (value - vmin) / (vmax - vmin)
    norm = np.clip(norm, 0, 1)

    # OpenCV colormap expects 0–255
    color_int = int(norm * 255)
    color = cv2.applyColorMap(np.uint8([[color_int]]), cv2.COLORMAP_JET)[0][0]
    return tuple(int(c) for c in color)  # (B, G, R)


def draw_colorbar(height=400, width=40, vmin=0.0, vmax=2.0):
    """Create a vertical colorbar image."""
    gradient = np.linspace(255, 0, height).astype(np.uint8).reshape(height, 1)
    gradient_img = np.repeat(gradient, width, axis=1)

    gradient_color = cv2.applyColorMap(gradient_img, cv2.COLORMAP_JET)

    # Add border & labels
    bar = gradient_color.copy()
    bar = cv2.copyMakeBorder(bar, 10, 10, 10, 50, cv2.BORDER_CONSTANT, value=(255, 255, 255))

    # Put labels
    cv2.putText(bar, f"{vmax:.1f}m", (50, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)
    cv2.putText(bar, f"{vmin:.1f}m", (50, height + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)

    return bar


def main(args):

    # Get Domain Map
    cv_image, map_metadata = get_domain_map(args.domain_config)

    # Read Poses from GT and 
    with open(args.ace_pose_file, "r") as f:
        ace_pose_data = f.readlines()

    # Dict mapping file name to ACE estimate
    ace_estimates = {}

    # parse pose file data
    for pose_line in ace_pose_data:
        file_name, pose_w2c, _, confidence = data_io.parse_ace_pose_line(pose_line)
        pose_c2w = np.linalg.inv(pose_w2c)
        ace_estimates[file_name] = (pose_c2w, confidence)

    sorted_ace_poses = [ace_estimates[key] for key in sorted(ace_estimates.keys())]
    print(f"Read {len(sorted_ace_poses)} estimate poses from: {args.ace_pose_file}")

    
    # # load ground truth poses, sorted by file name
    sorted_gt_poses = data_io.load_pose_files(args.gt_pose_files)
    print(f"Read {len(sorted_ace_poses)} gt poses from: {args.gt_pose_files}")

    # Filter Invalid Poses
    sorted_ace_poses = data_io.filter_invalid_poses(poses=sorted_gt_poses, items=sorted_ace_poses)
    sorted_gt_poses = data_io.filter_invalid_poses(poses=sorted_gt_poses, items=sorted_gt_poses)
    sorted_gt_poses = [pose.numpy() for pose in sorted_gt_poses]

    # We hardcode the alignement transformation for now, incase if we need to add that function
    alignment_transformation = np.eye(4)
    alignment_scale = 1.0

    x_offset = int(math.ceil(abs(map_metadata['origin'][0]) * 50))
    y_offset = cv_image.shape[0] - int(math.ceil(abs(map_metadata['origin'][1]) * 50))

    for (ace_pose, _), gt_pose in tqdm.tqdm(
        zip(sorted_ace_poses, sorted_gt_poses), dynamic_ncols=True, desc="Evaluating poses"
    ):
        if alignment_transformation is not None:
            # Apply alignment transformation to GT pose
            gt_pose = alignment_transformation @ gt_pose

            # Calculate translation error.
            t_err = float(np.linalg.norm(gt_pose[0:3, 3] - ace_pose[0:3, 3]))

            # Correct translation scale with the inverse alignment scale (since we align GT with estimates)
            t_err = t_err / alignment_scale

            # Rotation error.
            gt_R = gt_pose[0:3, 0:3]
            out_R = ace_pose[0:3, 0:3]

            r_err = np.matmul(out_R, np.transpose(gt_R))
            # Compute angle-axis representation.
            r_err = cv2.Rodrigues(r_err)[0]
            # Extract the angle.
            r_err = np.linalg.norm(r_err) * 180 / math.pi
        else:
            t_err, r_err = math.inf, math.inf
            continue

        # Draw on image
        trans = gt_pose[:3, 3]
        u = int(math.ceil(trans[1] * 50)) + x_offset
        v = int(math.ceil(-trans[2] * 50)) + y_offset

        color = map_value_to_color(t_err, vmax=args.vmax)
        cv2.circle(cv_image, (u, v), radius=4, color=color, thickness=-1)

    # Draw Color Bar
    colorbar = draw_colorbar(height=cv_image.shape[0], vmin=0.0, vmax=args.vmax)
    cb_resized = cv2.resize(colorbar, (colorbar.shape[1], cv_image.shape[0]))
    combined = np.hstack([cv_image, cb_resized])

    # Save
    cv2.imwrite(args.output_image, combined)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description='RestAPI inference on a specific scene.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument('ace_pose_file', type=Path, help='Path to an ACE pose file with one line per image')

    parser.add_argument('gt_pose_files', type=str,
                            help="Glob pattern for pose files, e.g. 'datasets/scene/*.txt', each file is "
                                 "assumed to contain a 4x4) pose matrix, cam2world")
   
    parser.add_argument('--output_image', type=Path, default="output.jpeg", help='Path to output image')

    parser.add_argument('--domain_config', type=Path, help='Path to a network trained for the scene (just the head weights)')

    parser.add_argument('--vmax', type=float, default=2.0, help='Max value of translation error for color map')

    args = parser.parse_args()

    main(args)