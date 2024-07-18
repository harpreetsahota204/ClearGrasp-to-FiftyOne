import os
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
import multiprocessing
import re
from glob import glob
from tqdm import tqdm

from utils import calculate_camera_parameters, load_image_cv2, exr_loader, write_point_cloud, load_masks_dict
import fiftyone as fo

def process_scene(args):
    """
    Process a single scene directory, generating point cloud (PCD) and FiftyOne 3D scene (FO3D) files.

    This function iterates through all frames in a scene, processes the RGB images, depth maps, 
    and mask data to create point clouds and 3D scenes. The resulting files are saved in the 
    specified output directory.

    Args:
        scene_dir (str): Path to the input scene directory. This directory should contain 
                         subdirectories for RGB images, depth maps, and JSON mask files.
        output_dir (str): Path to the output directory where PCD and FO3D files will be saved.

    The function expects the following file structure in the scene directory:
    - rgb-imgs/: Contains RGB image files named as '{frame_number:09d}-rgb.jpg'
    - depth-imgs-rectified/: Contains depth map files named as '{frame_number:09d}-depth-rectified.exr'
    - json-files/: Contains mask data files named as '{frame_number:09d}-masks.json'

    The function generates:
    - PCD files: Named as '{scene_name}_{frame_number:09d}.ply' in the output directory
    - FO3D files: Named as '{scene_name}_{frame_number:09d}.fo3d' in the output directory

    If any required file for a frame is missing, that frame is skipped with a warning message.

    """
    scene_dir, output_dir, i = args
    scene_name = os.path.basename(scene_dir).replace('-train', '')
    base_name = f"{i:09d}"

    image_path = os.path.abspath(os.path.join(scene_dir, "rgb-imgs", f"{base_name}-rgb.jpg"))
    depth_path = os.path.abspath(os.path.join(scene_dir, "depth-imgs-rectified", f"{base_name}-depth-rectified.exr"))
    masks_path = os.path.abspath(os.path.join(scene_dir, "json-files", f"{base_name}-masks.json"))

    if all(os.path.exists(path) for path in [image_path, depth_path, masks_path]):
        depth_array = exr_loader(depth_path, ndim=1)
        color_image_array = load_image_cv2(image_path)
        masks_dict = load_masks_dict(masks_path)

        try:
            # Try to calculate camera parameters using masks_dict
            fx, fy, cx, cy = calculate_camera_parameters(masks_dict)
        except KeyError:
            # If 'image' key is missing in masks_dict, use the fallback
            # Get image dimensions from color_image_array
            image_height, image_width = color_image_array.shape[:2]
            fx, fy, cx, cy = calculate_camera_parameters(
                masks_dict=masks_dict,
                image_width=image_width, 
                image_height=image_height
                )

        # Generate point cloud
        pcd_path = os.path.abspath(os.path.join(output_dir, f"{scene_name}_{base_name}.ply"))
        write_point_cloud(pcd_path, color_image_array, depth_array, fx, fy, cx, cy)

        #get quaternion
        q_X, q_Y, q_Z, q_W = masks_dict['camera']['world_pose']['rotation']['quaternion']
        quaternion = fo.Quaternion(x=q_X, y=q_Y, z=q_Z, w=q_W)

        # Create 3D scene
        scene = fo.Scene(fo.PerspectiveCamera(up="Z"))

        # instantiate mesh
        mesh = fo.PlyMesh(
            base_name,
            pcd_path,
            is_point_cloud=True
        )

        mesh.quaternion = quaternion

        #set material
        mesh.default_material = fo.PointCloudMaterial(shading_mode="rgb")

        # add to scene
        scene.add(mesh)

        fo3d_path = os.path.abspath(os.path.join(output_dir, f"{scene_name}_{base_name}.fo3d"))

        scene.write(fo3d_path)
        return True
    else:
        print(f"Skipping {base_name} due to missing files")
        return False

def process_scenes(scene_dir, output_dir):
    """
    Process a single scene directory, generating point cloud (PCD) and FiftyOne 3D scene (FO3D) files.
    """
    scene_name = os.path.basename(scene_dir).replace('-train', '')

    args_list = [(scene_dir, output_dir, i) for i in range(251)]

    with multiprocessing.Pool() as pool:
        results = list(tqdm(pool.imap(process_scene, args_list), total=251, desc=f"Processing {scene_name}"))

    return sum(results)  # Return the number of successfully processed frames


if __name__ == "__main__":
    base_dir = "/Users/harpreetsahota/workspace/ClearGrasp-to-FiftyOne/data/cleargrasp-dataset-train" 
    scene_dirs = sorted(glob(f"{base_dir}/*"))
    total_processed = 0  
    
    for scene_dir in scene_dirs:
        output_dir = f"{scene_dir}/point-clouds" 
        os.makedirs(output_dir, exist_ok=True)
        processed = process_scenes(scene_dir, output_dir)
        total_processed += processed

print("PCD and FO3D generation complete")