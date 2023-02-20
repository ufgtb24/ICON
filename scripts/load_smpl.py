import numpy as np
import trimesh
from scipy.spatial.transform import Rotation as R
import vedo
import open3d as o3d


# smpl_face=np.loadtxt('/home/yu/AMirror/ICON/data/cape_raw/00096/fit/face_idx.txt',dtype=np.int)
#
# minimal_body = np.load('/home/yu/AMirror/ICON/data/cape_raw/00096/fit/00096_minimal.npy')
#
# x = trimesh.Trimesh(
#     minimal_body, smpl_face, process=False, maintain_order=True)
# x.visual.vertex_colors = [128.0, 128.0, 0.0, 255.0]
#
#
# vp = vedo.Plotter(title="", size=(1500, 1500), axes=0, bg='white')
# vp.show(x, bg="white", axes=1.0, interactive=True)


##################
mesh_file='/home/yu/AMirror/ICON/data/cape_raw/00096/fit/00096_minimal.ply'
mesh = trimesh.load(mesh_file, file_type='ply')
ply_v = mesh.vertices
faces = mesh.faces
normals = mesh.vertex_normals
npy_file='/home/yu/AMirror/ICON/data/cape_raw/00096/fit/00096_minimal.npy'

npy_v = np.load(npy_file)


pcd1 = o3d.geometry.PointCloud()
pcd2 = o3d.geometry.PointCloud()

pcd1.points = o3d.utility.Vector3dVector(ply_v)
pcd2.points = o3d.utility.Vector3dVector(npy_v)

# 构建KDTree
tree = o3d.geometry.KDTreeFlann(pcd1)

# 对pcd1中的每个点寻找最近邻
idxs=np.zeros([6980],np.int32)
rank_idx=[]
for i,n in enumerate(npy_v):
    _, idx, _ = tree.search_knn_vector_3d(n, 1)
    idxs[idx[0]]=i
    rank_idx.append(idx[0])

face_npy=[[idxs[y] for y in x] for x in faces]
########## write it down
np.savetxt('/home/yu/AMirror/ICON/data/cape_raw/00096/fit/face_idx.txt',
           face_npy,fmt='%d')

np.savetxt('/home/yu/AMirror/ICON/data/cape_raw/00096/fit/rank_idx.txt',
           rank_idx,fmt='%d')

# # 重新排列pcd2的点
# pcd2_reordered = o3d.geometry.PointCloud()
# pcd2_reordered.points = o3d.utility.Vector3dVector(np.asarray(pcd1.points)[idxs])
#
# # 显示匹配后的点云
# o3d.visualization.draw_geometries([pcd1, pcd2_reordered])






pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(npy_v)
o3d.visualization.draw_geometries([pcd])


# pcd = o3d.geometry.PointCloud()
# pcd.points = o3d.utility.Vector3dVector(point_cloud)
# pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
# poisson_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=10, width=0, scale=1.1, linear_fit=False)[0]
# o3d.visualization.draw_geometries([poisson_mesh])


data_path= '/home/yu/AMirror/ICON/data/cape_raw/00096/fit/shortlong_hips.000001.npz'
points_dict = np.load(data_path)

pose_body = points_dict['pose_body'].astype(np.float32)
pose_hand = points_dict['pose_hand'].astype(np.float32)
trans = points_dict['trans'].astype(np.float32)
bone_transforms = points_dict['bone_transforms'].astype(np.float32)

pose = np.concatenate([pose_body, pose_hand], axis=-1)

pose = R.from_rotvec(pose.reshape([-1, 3]))

pose_mat = pose.as_matrix()
ident = np.eye(3)
pose_feature = (pose_mat - ident).reshape([207, 1])


posedir=dict(np.load('/home/yu/AMirror/ICON/data/cape_raw/00096/fit/posedirs_all.npz'))['male']
skinning_weights = dict(np.load('/home/yu/AMirror/ICON/data/cape_raw/00096/fit/skinning_weights_all.npz'))['male']
pose_offsets = np.dot(posedir.reshape([-1, 207]), pose_feature).reshape([6890, 3])

# smpl = trimesh.Trimesh(
#     minimal_shape, faces, process=False, maintain_order=True)
# smpl.visual.vertex_colors = [128.0, 128.0, 0.0, 255.0]

# vp = vedo.Plotter(title="", size=(1500, 1500), axes=0, bg='white')
# vp.show(smpl, bg="white", axes=1.0, interactive=True)

npy_v += pose_offsets
pcd.points = o3d.utility.Vector3dVector(npy_v)
o3d.visualization.draw_geometries([pcd])

#=============
smpl = trimesh.Trimesh(
    npy_v, face_npy, process=False, maintain_order=True)
smpl.visual.vertex_colors = [128.0, 128.0, 0.0, 255.0]
vp = vedo.Plotter(title="", size=(1500, 1500), axes=0, bg='white')
vp.show(smpl, bg="white", axes=1.0, interactive=True)
#=============


n_smpl_points=npy_v.shape[0]
homogen_coord = np.ones([n_smpl_points, 1], dtype=np.float32)

T = np.dot(skinning_weights, bone_transforms.reshape([-1, 16])).reshape([-1, 4, 4])
a_pose_homo = np.concatenate([npy_v, homogen_coord], axis=-1).reshape([n_smpl_points, 4, 1])
minimal_body_mesh = np.matmul(T, a_pose_homo)[:, :3, 0].astype(np.float32) + trans
# pose minimal-cloth
vis_list=[]

smpl = trimesh.Trimesh(
    minimal_body_mesh, face_npy, process=False, maintain_order=True)
smpl.visual.vertex_colors = [128.0, 128.0, 0.0, 255.0]


vp = vedo.Plotter(title="", size=(1500, 1500), axes=0, bg='white')
vp.show(smpl, bg="white", axes=1.0, interactive=True)



