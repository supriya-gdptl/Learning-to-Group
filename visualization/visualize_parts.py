# coding: utf-8


def visualize_trimesh_pointcloud():
    import h5py
    import trimesh
    import numpy as np


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


def save_open3d_images_of_pointcloud():
    """
    save renderings of an object with bounding box, coordinate axes
    It saves the images from three different views
    :param mesh_path: mesh or point cloud path
    :param output_image_name: name of the rendered image to save to
    :return:
    """

    import h5py
    import numpy as np
    import open3d as o3d
    import copy

    o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Warning)
    points_path = "../data/partnet/ins_seg_h5_gt/Bag-1/test-00.h5"
    predicted_mask_path = "../pretrained_results/Bag/Level_2/test-00.h5"
    # import os
    # print(os.path.abspath(points_path))
    # print(os.path.abspath(predicted_mask_path))
    pts = h5py.File(points_path, mode='r')['pts'][...]
    # mask size=(num_examples, 200, 10k)
    mask = h5py.File(predicted_mask_path, mode='r')['mask'][...]
    # valid size = (num_examples, 200). Boolean array
    valid = h5py.File(predicted_mask_path, mode='r')['valid'][...]

    # point cloud index
    pcd_idx = 0

    pcd_mask = mask[pcd_idx]
    # get valid masks out of 200. ('mask' contains 200 parts, but only some of them represent parts, others are empty)
    # array/channel ids of such non-empty parts is given in 'valid'
    pcd_valid_part_ids = np.where(valid[pcd_idx]==True)[0]

    # save images
    dirs = (np.pi/8, -np.pi/3, -np.pi/8)

    for part_idx in pcd_valid_part_ids:
        pts0 = pts[pcd_idx]

        mask0 = np.expand_dims(pcd_mask[part_idx], axis=1)

        # get part coordinates
        part = np.multiply(pts0, mask0)

        # convert numpy array to open3d point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(part)

        # create transformation matrix
        T = np.eye(4)
        T[:3, :3] = pcd.get_rotation_matrix_from_xyz(dirs)
        pcd_t = copy.deepcopy(pcd).transform(T)

        # box = pcd_t.get_oriented_bounding_box()
        # box.color = (1, 0, 0)

        vis = o3d.visualization.Visualizer()
        vis.create_window(visible=False, width=256, height=256, left=0, top=0)
        vis.add_geometry(pcd_t)
        # vis.add_geometry(box)

        vis.get_render_option().load_from_json('renderoption.json')
        vis.update_geometry(pcd_t)
        vis.poll_events()
        vis.update_renderer()
        vis.capture_screen_image(f"bag_level2_img_{pcd_idx}_part_{part_idx}.png")
        vis.destroy_window()


if __name__ == '__main__':
    save_open3d_images_of_pointcloud()

