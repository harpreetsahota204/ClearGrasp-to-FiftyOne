import os
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'

import re
from glob import glob
from tqdm import tqdm

from utils import calculate_camera_parameters, load_image_cv2, exr_loader, write_point_cloud, load_masks_dict
import fiftyone as fo

def process_scene(scene_dir, output_dir):
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
    scene_name = os.path.basename(scene_dir).replace('-train', '')
    
    for i in tqdm(range(101), desc=f"Processing {scene_name}"):
        base_name = f"{i:09d}"  # This will generate '000000000', '000000001', etc.

        image_path = os.path.join(scene_dir, "rgb-imgs", f"{base_name}-rgb.jpg")
        depth_path = os.path.join(scene_dir, "depth-imgs-rectified", f"{base_name}-depth-rectified.exr")
        masks_path = os.path.join(scene_dir, "json-files", f"{base_name}-masks.json")

        if all(os.path.exists(path) for path in [image_path, depth_path, masks_path]):
            masks_dict = utils.load_masks_dict(masks_path)
            fx, fy, cx, cy = calculate_camera_parameters(masks_dict)

            depth_array = utils.exr_loader(depth_path, ndim=1)
            color_image_array = utils.load_image_cv2(image_path)

            # Generate point cloud
            pcd_path = os.path.join(output_dir, f"{scene_name}_{base_name}.ply")
            utils.write_point_cloud(pcd_path, color_image_array, depth_array, fx, fy, cx, cy)

            #get quaternion
            q_X = masks_dict['camera']['world_pose']['rotation']['quaternion'][0]
            q_Y = masks_dict['camera']['world_pose']['rotation']['quaternion'][1]
            q_Z = masks_dict['camera']['world_pose']['rotation']['quaternion'][2]
            q_W = masks_dict['camera']['world_pose']['rotation']['quaternion'][3]

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

            fo3d_path = os.path.join(output_dir, f"{scene_name}_{base_name}.fo3d")

            scene.write(fo3d_path)
        else:
            print(f"Skipping {base_name} due to missing files")

# Main execution
if __name__ == "__main__":
    base_dir = "./data/cleargrasp-dataset-train"  # Replace with your dataset path
    output_dir = f"{base_dir}/point-clouds"  # Replace with your desired output path
    os.makedirs(output_dir, exist_ok=True)

    scene_dirs = glob(f"{base_dir}/*")
    for scene_dir in scene_dirs:
        process_scene(os.path.join(base_dir, scene_dir), output_dir)

print("PCD and FO3D generation complete")