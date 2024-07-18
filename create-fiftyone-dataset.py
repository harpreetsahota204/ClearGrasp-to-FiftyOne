import os
import pandas as pd
import fiftyone as fo
import fiftyone.core.fields as fof
from typing import Dict, List, Any

def create_dataset(name) -> fo.Dataset:
    """
    Creates schema for a FiftyOne dataset.
    """

    dataset = fo.Dataset(
        name=name,
        persistent=True,
        overwrite=True
    )

    dataset.add_group_field("group", default="rgb")

    return dataset

def create_fo_sample(media: dict, dataset: fo.Dataset) -> fo.Sample:
    """
    Creates a FiftyOne Sample from a given image entry with metadata and custom fields.

    Args:
        media (dict): A dictionary containing media data including the path and other properties.

    Returns:
        fo.Sample: The FiftyOne Sample object with the image and its metadata.
    """
    samples = []

    for scene_dir in scene_dirs:
        # Get all image files from scene directory
        image_paths = sorted(glob(f"{scene_dir}/rgb-imgs/*"))

        for i, image_path in enumerate(image_paths):
            # Extract the base name (e.g., '000000000-rgb')
            base_name = os.path.splitext(os.path.basename(image_path))[0]

            # Remove the '-rgb' suffix for other file types
            base_name_without_suffix = base_name.replace('-rgb', '')

            # Construct paths for corresponding files
            depth_path = f"{scene_dir}/depth-imgs-rectified/{base_name_without_suffix}-depth-rectified.exr"
            masks_path = f"{scene_dir}/segmentation-masks/{base_name_without_suffix}-segmentation-mask.png"
            outlines_path = f"{scene_dir}/outlines/{base_name_without_suffix}-outlineSegmentation.png"

            # Check if all required files exist
            if all(os.path.exists(path) for path in [depth_path, masks_path, outlines_path]):
                depth_map = cv2.imread(depth_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
                depth_map = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX)
                depth_map = depth_map.astype("uint8")

                #split mask by instances
                segs = fo.Segmentation(mask_path=masks_path).to_polylines(mask_types="thing", tolerance=1)['polylines']
                for lines in segs:
                lines.set_field("label", re.sub(r'-train$', '', os.path.basename(scene_dir)))

                sample = fo.Sample(
                    filepath=image_path,
                    gt_depth=fo.Heatmap(map=depth_map),
                    gt_segmentation_mask=fo.Polylines(polylines=segs),
                    gt_outline=fo.Segmentation(mask_path=outlines_path),
                )

                samples.append(sample)
            else:
                print(f"Skipping {base_name} due to missing files")
                print(f"  Image path: {image_path}")
                print(f"  Depth path: {depth_path}")
                print(f"  Masks path: {masks_path}")
                print(f"  Outlines path: {outlines_path}")


    add_samples_to_fiftyone_dataset(dataset, samples)

def add_samples_to_fiftyone_dataset(
    dataset: fo.Dataset,
    samples: list
    ):
    """
    Creates a FiftyOne dataset from a list of samples.

    Args:
      samples (list): _description_
      dataset_name (str): _description_
    """
    dataset.add_samples(samples)

if __name__ == "__main__":
    DATA_DIR = ""
    DATASET_NAME = "WayveScenes101"


    scene_metadata_filename = []
    for i, scene_dict in enumerate(scene_metadata):
        scene_id = f"scene_{str(i+1).zfill(3)}"
        video_file = os.path.join(DATA_DIR, scene_id)

        # Check if the video file exists
        if os.path.exists(video_file):
            scene_dict = scene_dict.copy()  # Create a copy to avoid modifying the original

            # Get the directory and base filename without extension
            video_dir = os.path.dirname(video_file)
            video_base = os.path.splitext(os.path.basename(video_file))[0]

            scene_dict["file_paths"] = {
                "front_forward": os.path.join(video_dir, scene_id, f"{video_base}_front-forward.mp4"),
                "left_backward": os.path.join(video_dir, scene_id, f"{video_base}_left-backward.mp4"),
                "left_forward": os.path.join(video_dir, scene_id, f"{video_base}_left-forward.mp4"),
                "right_backward": os.path.join(video_dir, scene_id, f"{video_base}_right-backward.mp4"),
                "right_forward": os.path.join(video_dir, scene_id, f"{video_base}_right-forward.mp4"),
                "point_cloud": os.path.join(video_dir, scene_id, f"{video_base}.fo3d")
            }
            scene_metadata_filename.append(scene_dict)

    print(f"Total scenes with existing video files: {len(scene_metadata_filename)}")

    # Create the FiftyOne dataset
    dataset = create_dataset(DATASET_NAME)

    for video in scene_metadata_filename:
        create_fo_sample(video, dataset)
    
    dataset.compute_metadata()
    dataset.save()

    print(f"Created dataset '{DATASET_NAME}' with {len(dataset)} samples")
