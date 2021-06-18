# coding: utf-8
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
