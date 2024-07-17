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

    sample_fields = {field: media.get(value) for field, value in fields.items()}

    filepaths = media.get("file_paths")

    samples = []

    group = fo.Group()

    for name, filepath in filepaths.items():
      sample = fo.Sample(
          filepath=filepath,
          group=group.element(name),
          **sample_fields)
      samples.append(sample)

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
    DATA_DIR = "/Users/harpreetsahota/workspace/datasets/wayve_101"
    DATASET_NAME = "WayveScenes101"
    METADATA_CSV = os.path.join(DATA_DIR, "scene_metadata.csv")

    # Read the CSV file
    scene_metadata_df = pd.read_csv(METADATA_CSV)
    scene_metadata = scene_metadata_df.to_dict(orient="records")

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
