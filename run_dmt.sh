model_name="ace_g_25min"
dmt_dataset="/workspace/data/datasets/10F-CV-Test/job_7d145d28-11ce-4638-be24-5eafc2f1dbaa"
dmt_dataset_name="10F-CV-Test"

# Setting up dmt dataset
python3 datasets/setup_dmt.py --dmt_path "${dmt_dataset}"

python datasets/setup_colmap.py \
    --colmap_path "${dmt_dataset}/refined/global/refined_sfm_combined" \
    --raw_image_path "${dmt_dataset}/Frames" \
    --output_path "${dmt_dataset}/frames_colmap" \
    --global_setup True

# Visualize mapping on mapping sequence
echo "Running mapping..."
python -m ace_g.train_single_scene \
    --config "${model_name}.yaml" \
    --dataset.rgb_files "${dmt_dataset}/global_setup/train/rgb/*.jpg" \
    --dataset.pose_files "${dmt_dataset}/global_setup/train/poses/*.txt" \
    --dataset.calibration_source "heuristic" \
    --calibration_source "heuristic" \
    --session_id "${model_name}-${dmt_dataset_name}" \
    --rerun_spawn False \
    --use_rerun True

# Visualize localization on query sequence
echo "Running localization..."
python -m ace_g.vis_localization \
    --config "./outputs/${model_name}-${dmt_dataset_name}_map.yaml" \
    --dataset.rgb_files "${dmt_dataset}/global_setup/test/rgb/*.jpg" \
    --dataset.pose_files "${dmt_dataset}/global_setup/test/poses/*.txt" \
    --dataset.calibration_source "heuristic" \
    --calibration_source "heuristic" \
    --rerun_spawn False \
    --use_rerun True

# Evaluating result
python -m ace_g.eval_poses \
  --config "./outputs/${model_name}-${dmt_dataset_name}_reg.yaml" \
  --gt_pose_files "${dmt_dataset}/global_setup/test/poses/*.txt"