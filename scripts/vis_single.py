import sys

import numpy as np

sys.path.append('/home/yu/AMirror/ICON')
# generate visible mask of smpl vertex for different calib(rotation)
from lib.dataset.mesh_util import projection, load_calib, get_visibility
from lib.renderer.mesh import load_fit_body, load_smpl_body
import argparse
import os
import time
import trimesh
import torch
import glob

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
    '-m', '--mode', type=str, help='gen/debug')

args = parser.parse_args()

subject = args.subject
seq = args.seq
frame = args.frame

save_folder = args.out_dir
rotation = int(args.rotation)

dataset = save_folder.split("/")[2]
frame_num=frame.split('/')[-1]


mesh_file = f'./data/{dataset}/{subject}/scans_ply/{seq}/{seq}.{frame_num}.ply'
fit_file = f'./data/{dataset}/{subject}/fit/{seq}/{seq}.{frame_num}.npz'



mesh = trimesh.load(mesh_file, file_type='ply')

# remove floating outliers of scans
mesh_lst = mesh.split(only_watertight=False)
comp_num = [mesh.vertices.shape[0] for mesh in mesh_lst]
mesh = mesh_lst[comp_num.index(max(comp_num))]

vertices = mesh.vertices / 1000.
faces = mesh.faces

smpl_vert, _ = load_smpl_body(fit_file,100.)
smpl_face=np.loadtxt('/home/yu/AMirror/ICON/data/cape_raw/00096/fit/face_idx.txt',dtype=np.int32)



smpl_verts = torch.from_numpy(smpl_vert).cuda().float()
smpl_faces = torch.from_numpy(smpl_face).cuda().long()

for y in range(0, 360, 360//rotation):
    calib_file = os.path.join(
        save_folder, frame_num, 'calib', f'{y:03d}.txt')

    vis_file = os.path.join(
        save_folder, frame_num, 'vis', f'{y:03d}.txt')

    os.makedirs(os.path.dirname(vis_file), exist_ok=True)

    if not os.path.exists(vis_file):

        calib = load_calib(calib_file).cuda()
        calib_verts = projection(smpl_verts, calib, format='tensor')
        (xy, z) = calib_verts.split([2, 1], dim=1)
        smpl_vis = get_visibility(xy, z, smpl_faces)

        if args.mode == 'debug':
            mesh = trimesh.Trimesh(smpl_verts.cpu().numpy(
            ), smpl_faces.cpu().numpy(), process=False)
            mesh.visual.vertex_colors = torch.tile(smpl_vis, (1, 3)).numpy()
            mesh.export(vis_file.replace("pt", "obj"))

        torch.save(smpl_vis, vis_file)

# done_jobs = len(glob.glob(f"{save_folder}/{frame_num}/*/vis"))
# all_jobs = len(os.listdir(f"./data/{dataset}/scans"))
# print(
#     f"Finish visibility computing {subject}| {done_jobs}/{all_jobs} | Time: {(time.time()-t0):.0f} secs")
