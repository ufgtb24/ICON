import sys

import vedo

sys.path.append('/home/yu/AMirror/ICON')
# generate renderimage and normal from different cam rotation, and record normalization and rotation to calib
import lib.renderer.opengl_util as opengl_util
from lib.renderer.mesh import load_fit_body, load_scan, compute_tangent, load_ori_fit_body, load_smpl_body, gen_uv
import lib.renderer.prt_util as prt_util
from lib.renderer.gl.init_gl import initialize_GL_context
from lib.renderer.gl.prt_render import PRTRender
from lib.renderer.gl.color_render import ColorRender
from lib.renderer.camera import Camera
import argparse
import os
import glob
import cv2
import numpy as np
import random
import math
import time
import trimesh
from matplotlib import cm
import open3d as o3d



pid=os.getpid()
print(pid)
input()
# import sys
t0 = time.time()

parser = argparse.ArgumentParser()
parser.add_argument(
    '-s', '--subject', type=str, help='subject name')
parser.add_argument(
    '-q', '--seq', type=str, help='subject name')
parser.add_argument(
    '-f', '--frame', type=str, help='subject name')
parser.add_argument(
    '-o', '--out_dir', type=str, help='output dir')
parser.add_argument(
    '-r', '--rotation', type=str, help='rotation num')
parser.add_argument(
    '-w', '--size', type=str, help='render size')
args = parser.parse_args()

subject = args.subject
seq = args.seq
frame = args.frame
save_folder = args.out_dir
rotation = int(args.rotation)
size = int(args.size)
frame_num=frame.split('.')[1]
# headless
egl = True

# render
initialize_GL_context(width=size, height=size, egl=egl)

dataset = save_folder.split("/")[2]

scale = 100.0
up_axis = 1
pcd = True
smpl_type = "smplx"
with_light = True
depth = False
normal = True

# print(f'render pid : {os.getpid()}')
# time.sleep()

mesh_file = f'./data/{dataset}/{subject}/scans_ply/{seq}/{frame}'
smplx_file = f'./data/{dataset}/{subject}/smplx/{seq}/{frame}.obj'
# tex_file = f'./data/{dataset}/scans/{subject}/material0.jpeg'
fit_file = f'./data/{dataset}/{subject}/fits/{seq}/{frame}/smplx_param.pkl'

# mesh
# mesh = trimesh.load(mesh_file, skip_materials=False,
#                     process=False, maintain_order=True, force='mesh',file_type='ply')
# mesh = trimesh.load(trimesh.util.wrap_as_stream(mesh_file), skip_materials=False,
#                     process=False, maintain_order=True, force='mesh',file_type='ply')

if not pcd:
    # print("Hello !!!!!!!!!!!!!!!")
    
    vertices, faces, normals, faces_normals, textures, face_textures = load_scan(
        mesh_file, with_normal=True, with_texture=True)
    
else:
    mesh = trimesh.load(mesh_file, file_type='ply')
    
    # remove floating outliers of scans
    mesh_lst = mesh.split(only_watertight=False)
    comp_num = [mesh.vertices.shape[0] for mesh in mesh_lst]
    mesh = mesh_lst[comp_num.index(max(comp_num))]

    vertices = mesh.vertices
    faces = mesh.faces
    normals = mesh.vertex_normals
    face_textures=faces
    faces_normals=faces
    colors = mesh.visual.vertex_colors[:, :3] / 255.0
    textures,texture_image=gen_uv(colors, texture_size=(5000, 5000))

# center

scan_scale = 1.8/(vertices.max(0)[up_axis] - vertices.min(0)[up_axis])

smpl_face=np.loadtxt('/home/yu/AMirror/ICON/data/cape_raw/00096/fit/face_idx.txt',dtype=np.int)
# rank_idx=np.loadtxt('/home/yu/AMirror/ICON/data/cape_raw/00096/fit/rank_idx.txt',dtype=np.int)
# smpl_normals = trimesh.load(mesh_file, file_type='ply')


smpl_vert, joints = load_smpl_body('/home/yu/AMirror/ICON/data/cape_raw/00096/fit/shortlong_hips.000001.npz',scale)
# rescale_fitted_body, joints = load_fit_body(fit_file,
#                                             scale,
#                                             smpl_type='smplx',
#                                             smpl_gender='male')



rescale_fitted_body = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(smpl_vert), o3d.utility.Vector3iVector(smpl_face))
# 计算顶点法向量
rescale_fitted_body.compute_vertex_normals()

# x = trimesh.Trimesh(
#     np.asarray(rescale_fitted_body.vertices), smpl_face, process=False, maintain_order=True)
# x.visual.vertex_colors = [128.0, 128.0, 0.0, 255.0]
#
#
# vp = vedo.Plotter(title="", size=(1500, 1500), axes=0, bg='white')
# vp.show(x, bg="white", axes=1.0, interactive=True)

# os.makedirs(os.path.dirname(smplx_file), exist_ok=True)
# ori_smplx = load_ori_fit_body(fit_file,
#                               smpl_type='smplx',
#                               smpl_gender='male')
# ori_smplx.export(smplx_file)

vertices *= scale
vmin = vertices.min(0)
vmax = vertices.max(0)
vmed = joints[0]
vmed[up_axis] = 0.5*(vmax[up_axis] + vmin[up_axis])

rndr_depth = ColorRender(width=size, height=size, egl=egl)
rndr_depth.set_mesh(np.asarray(rescale_fitted_body.vertices),
                    np.asarray(rescale_fitted_body.triangles),
                    np.asarray(rescale_fitted_body.vertices),
                    np.asarray(rescale_fitted_body.vertex_normals))
rndr_depth.set_norm_mat(scan_scale, vmed)


# camera
cam = Camera(width=size, height=size)
cam.ortho_ratio = 0.4 * (512 / size)

# if pcd:
#
#     colors = mesh.visual.vertex_colors[:, :3] / 255.0
#     rndr = ColorRender(width=size, height=size, egl=egl)
#     rndr.set_mesh(vertices, faces, colors, normals)
#     rndr.set_norm_mat(scan_scale, vmed)
#     shs = np.load('./scripts/env_sh.npy')
#
# else:

prt, face_prt = prt_util.computePRT(mesh_file, scale, 10, 2)
shs = np.load('./scripts/env_sh.npy')
rndr = PRTRender(width=size, height=size, ms_rate=16, egl=egl)

# texture
# texture_image = cv2.imread(tex_file)
# texture_image = cv2.cvtColor(texture_image, cv2.COLOR_BGR2RGB)

# fake multiseg
vertices_label_mode = np.random.randint(low=1, high=10, size=(
    vertices.shape[0], 10))   # [scan_verts_n, percomp]
colormap = cm.get_cmap("rainbow")
precomp_id = 4
verts_label = colormap(vertices_label_mode[:, precomp_id]/np.max(
    vertices_label_mode[:, precomp_id]))[:, :3]  # [scan_verts_num, 3]

tan, bitan = compute_tangent(
    vertices, faces, normals, None, None)

rndr.set_norm_mat(scan_scale, vmed)
rndr.set_mesh(vertices, faces, normals, faces_normals,
              textures, face_textures,
              prt, face_prt, tan, bitan, verts_label)
rndr.set_albedo(texture_image)


for y in range(0, 360, 360//rotation):

    cam.near = -100
    cam.far = 100
    cam.sanity_check()

    R = opengl_util.make_rotate(0, math.radians(y), 0)
    R_B = opengl_util.make_rotate(0, math.radians((y+180) % 360), 0)

    if up_axis == 2:
        R = np.matmul(R, opengl_util.make_rotate(math.radians(90), 0, 0))

    rndr.rot_matrix = R
    rndr.set_camera(cam)

    if smpl_type != "none":
        rndr_depth.rot_matrix = R
        rndr_depth.set_camera(cam)

    dic = {'ortho_ratio': cam.ortho_ratio,
           'scale': scan_scale,
           'center': vmed,
           'R': R}

    if with_light:

        # random light
        sh_id = random.randint(0, shs.shape[0]-1)
        sh = shs[sh_id]
        sh_angle = 0.2*np.pi*(random.random()-0.5)
        sh = opengl_util.rotateSH(
            sh, opengl_util.make_rotate(0, sh_angle, 0).T)
        dic.update({"sh": sh})

        rndr.set_sh(sh)
        rndr.analytic = False
        rndr.use_inverse_depth = False

    # ==================================================================

    # calib
    calib = opengl_util.load_calib(dic, render_size=size)

    export_calib_file = os.path.join(
        save_folder, frame_num, 'calib', f'{y:03d}.txt')
    os.makedirs(os.path.dirname(export_calib_file), exist_ok=True)
    np.savetxt(export_calib_file, calib)

    # ==================================================================

    # front render
    rndr.display()

    opengl_util.render_result(rndr, 0, os.path.join(
        save_folder, frame_num, 'render', f'{y:03d}.png'))
    if normal:
        opengl_util.render_result(rndr, 1, os.path.join(
            save_folder, frame_num, 'normal_F', f'{y:03d}.png'))

    if depth:
        opengl_util.render_result(rndr, 2, os.path.join(
            save_folder, frame_num, 'depth_F', f'{y:03d}.png'))

    if smpl_type != "none":
        rndr_depth.display()
        opengl_util.render_result(rndr_depth, 1, os.path.join(
            save_folder, frame_num, 'T_normal_F', f'{y:03d}.png'))
        if depth:
            opengl_util.render_result(rndr_depth, 2, os.path.join(
                save_folder, frame_num, 'T_depth_F', f'{y:03d}.png'))

    # ==================================================================

    # back render
    cam.near = 100
    cam.far = -100
    cam.sanity_check()

    rndr.set_camera(cam)
    rndr.display()

    if normal:
        opengl_util.render_result(rndr, 1, os.path.join(
            save_folder, frame_num, 'normal_B', f'{y:03d}.png'))
    if depth:
        opengl_util.render_result(rndr, 2, os.path.join(
            save_folder, frame_num, 'depth_B', f'{y:03d}.png'))

    if smpl_type != "none":
        rndr_depth.set_camera(cam)
        rndr_depth.display()
        opengl_util.render_result(rndr_depth, 1, os.path.join(
            save_folder, frame_num, 'T_normal_B', f'{y:03d}.png'))
        if depth:
            opengl_util.render_result(rndr_depth, 2, os.path.join(
                save_folder, frame_num, 'T_depth_B', f'{y:03d}.png'))


# done_jobs = len(glob.glob(f"{save_folder}/*/render"))
# all_jobs = len(os.listdir(f"./data/{dataset}/scans"))
# print(
#     f"Finish rendering {subject}| {done_jobs}/{all_jobs} | Time: {(time.time()-t0):.0f} secs")
