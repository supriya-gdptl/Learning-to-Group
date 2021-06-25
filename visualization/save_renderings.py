# coding: utf-8
import h5py
import numpy as np
import open3d as o3d
import copy
import os
from colormap import colormap



def visualize_trimesh_pointcloud():
    import trimesh

    a=h5py.File("Chair-01/train-04.h5", mode='r')
    all_mask = a['gt_mask'][...]
    all_pts = a['pts'][...]
    pts0 = all_pts[0]
    print("0th point cloud shape: ", pts0.shape)

    # get zeroth mask
    mask0 = all_mask[0][0]
    # check if there are any '1's in the mask
    ones = np.where(mask0==1)
    print("number of 1 in mask: ",ones[0].shape)
    print("mask0 shape: ",mask0.shape)

    # display point cloud of original shape
    pcd = trimesh.points.PointCloud(pts0)
    pcd.show()

    # masks are of type 'boolean'. convert them to float to perform elementwise-multiplication
    mask0_float = mask0.astype("float")
    mask0_float = np.expand_dims(mask0_float, axis=1)

    mask0_float = mask0.astype(np.float32)

    # elementwise multiply mask and original shape to get mask
    part = np.multiply(pts0, mask0_float)
    print("part array shape: ",part.shape)

    # create point cloud object from numpy array
    part_pcd = trimesh.points.PointCloud(part)
    part_pcd.show()


def save_image(point_cloud, output_filename):
    """

    :param point_cloud:
    :param output_filename:
    :return:
    """
    # viewing direction
    dirs = (np.pi/6, np.pi/6, 0)

    # create transformation matrix
    T = np.eye(4)
    T[:3, :3] = point_cloud.get_rotation_matrix_from_xyz(dirs)
    pcd_t = copy.deepcopy(point_cloud).transform(T)

    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False, width=256, height=256, left=0, top=0)
    vis.add_geometry(pcd_t)

    vis.get_render_option().load_from_json('./visualization/renderoption.json')
    vis.update_geometry(pcd_t)
    vis.poll_events()
    vis.update_renderer()
    vis.capture_screen_image(f"{output_filename}")
    vis.destroy_window()


def save_open3d_images_of_pointcloud(category, predicted_mask_basepath, main_output_folder):
    """
    save renderings of an object and predicted parts
    :param category: (string) shape category
    :param predicted_mask_basepath:
    :param main_output_folder: (string) folder location to save output renderings
    :return:
    """
    # get list of colors
    color_list = colormap()  # (79,3) colors array
    # print("color_list.shape: ",color_list.shape)

    o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Warning)
    # keeping the level to '1' for points path, becoz we care only about the points and all categories have at least level-1 folder
    points_path = f"./data/partnet/ins_seg_h5_gt/{category}-1/test-00.h5"
    pts = h5py.File(points_path, mode='r')['pts'][...]

    # save images for all granularity. level of part segmentation granularity (level=1: coarser, level=3: finer)
    for level in [1]:
        # create output folder
        output_folder = os.path.join(main_output_folder, category, f"Level_{level}")
        print("output_folder:", os.path.abspath(output_folder))
        os.makedirs(output_folder, exist_ok=True)

        # open predicted part segmentation file
        predicted_mask_path = f"{predicted_mask_basepath}/{category}/Level_{level}/test-00.h5"
        predicted_mask_data = h5py.File(predicted_mask_path, mode='r')

        # save part segmented image for all shapes. save at max 100 images
        for point_cloud_idx in range(min(pts.shape[0], 100)):

            pts0 = pts[point_cloud_idx]
            # save original point cloud for reference
            # pcd = o3d.geometry.PointCloud()
            # pcd.points = o3d.utility.Vector3dVector(pts0)
            # pcd.paint_uniform_color(np.array([[0.0],[0.0],[0.0]]))
            # save_image(point_cloud=pcd, output_filename=os.path.join(output_folder, f"img_{category}_L{level}_pcd{point_cloud_idx}.png"))

            # mask size=(num_examples, 200, 10k)
            mask = predicted_mask_data['mask'][...]
            # valid size = (num_examples, 200). Boolean array
            valid = predicted_mask_data['valid'][...]

            # get predicted all masks of current point cloud
            pcd_mask = mask[point_cloud_idx]
            # get valid masks out of 200. ('mask' contains 200 parts, but only some of them represent parts, others are empty)
            # array/channel ids of such non-empty parts is given in 'valid'
            pcd_valid_part_ids = np.where(valid[point_cloud_idx]==True)[0]
            # Colors array of size (part_ids.shape, 3,1)
            pcd_color_list = color_list[:pcd_valid_part_ids.shape[0]]

            # collect part point cloud and colors
            part_pcd = np.zeros((10000, 3))
            part_color = np.zeros((10000, 3))

            for part_idx in pcd_valid_part_ids:
                mask0 = np.expand_dims(pcd_mask[part_idx], axis=1)

                # get part coordinates
                part = np.multiply(pts0, mask0)
                part_pcd += part

                # color
                color = np.tile(pcd_color_list[part_idx],(part.shape[0],1))
                color = np.multiply(color, mask0)
                part_color += color

            # convert numpy array to open3d point cloud
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(part_pcd)
            # use "paint_uniform_color(self: open3d.geometry.PointCloud, color: numpy.ndarray[float64[3, 1]]) → None" to assign colors to current part-pointcloud
            pcd.colors = o3d.utility.Vector3dVector(part_color)

            save_image(point_cloud=pcd, output_filename=os.path.join(output_folder, f"img_{category}_L{level}_pcd{point_cloud_idx}_segmented.png"))
            if (point_cloud_idx != 0 and point_cloud_idx%10 == 0) or point_cloud_idx == pts.shape[0]-1:
                print(f"Category: {category} - Level {level}: saved {point_cloud_idx} images")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--category", type=str, default='Bed', help="PartNet category")
    parser.add_argument("--main_output_folder", type=str, default="../../www/partnet_renderings/pretrained",
                        help="folder path to save images")
    parser.add_argument("--predicted_mask_basepath", type=str, default="./pretrained_results",
                        help="folder where predicted part label h5 files are saved")
    opt = parser.parse_args()
    save_open3d_images_of_pointcloud(category=opt.category,
                                     predicted_mask_basepath = opt.predicted_mask_basepath,
                                     main_output_folder=opt.main_output_folder)


