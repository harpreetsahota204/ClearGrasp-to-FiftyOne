import os
import pandas as pd
import fiftyone as fo
import fiftyone.core.fields as fof
from typing import Dict, List, Any
from glob import glob 
import cv2
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'

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

def create_fo_sample(scene_dir: str, dataset: fo.Dataset) -> fo.Sample:
    """
    Creates a FiftyOne Sample from a given image entry with metadata and custom fields.

    Args:
        media (dict): A dictionary containing media data including the path and other properties.

    Returns:
        fo.Sample: The FiftyOne Sample object with the image and its metadata.
    """
    
    # Get all image files from scene directory
    image_paths = sorted(glob(f"{scene_dir}/rgb-imgs/*"))
    scene_name = os.path.basename(scene_dir).replace('-train', '')

    samples = []

    for i, image_path in enumerate(image_paths):
        # Extract the base name (e.g., '000000000-rgb')
        base_name = os.path.splitext(os.path.basename(image_path))[0]

        # Remove the '-rgb' suffix for other file types
        base_name_without_suffix = base_name.replace('-rgb', '')

        # Construct paths for corresponding files
        depth_path = f"{scene_dir}/depth-imgs-rectified/{base_name_without_suffix}-depth-rectified.exr"
        masks_path = f"{scene_dir}/segmentation-masks/{base_name_without_suffix}-segmentation-mask.png"
        outlines_path = f"{scene_dir}/outlines/{base_name_without_suffix}-outlineSegmentation.png"
        pcd_path = f"{scene_dir}/point-clouds/{scene_name}_{base_name_without_suffix}.fo3d"

        # Check if all required files exist
        if all(os.path.exists(path) for path in [depth_path, masks_path, outlines_path, pcd_path]):
            depth_map = cv2.imread(depth_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
            depth_map = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX)
            depth_map = depth_map.astype("uint8")

            #split mask by instances
            segs = fo.Segmentation(mask_path=masks_path).to_polylines(mask_types="thing", tolerance=1)['polylines']
            for lines in segs:
                lines.set_field("label", scene_name)

            group = fo.Group()

            rgb_sample = fo.Sample(
                filepath=image_path,
                gt_depth=fo.Heatmap(map=depth_map),
                gt_segmentation_mask=fo.Polylines(polylines=segs),
                gt_outline=fo.Segmentation(mask_path=outlines_path),
                group=group.element("rgb")
            )

            pcd_sample = fo.Sample(
                filepath=pcd_path,
                group=group.element("point_cloud")
            )

            samples.append(rgb_sample)
            samples.append(pcd_sample)

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
    DATA_DIR = "/Users/harpreetsahota/workspace/ClearGrasp-to-FiftyOne/data/cleargrasp-dataset-train"

    DATASET_NAME = "ClearGrasp"

    OBJECT_DIRS = sorted(glob(f"{DATA_DIR}/*"))[:10]

    # Create the FiftyOne dataset
    dataset = create_dataset(DATASET_NAME)

    for sub_dir in OBJECT_DIRS:
        create_fo_sample(sub_dir, dataset)

    dataset.compute_metadata()
    dataset.save()

    print(f"Created dataset '{DATASET_NAME}' with {len(dataset)} samples")
