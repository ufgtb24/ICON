import pywavefront
import os
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument(
    '-s', '--subject', type=str, help='subject name')
parser.add_argument(
    '-q', '--seq', type=str, help='subject name')
parser.add_argument(
    '-f', '--frame', type=str, help='subject name')

args = parser.parse_args()
subject = args.subject
seq = args.seq
frame = args.frame

ply_path = f'./data/cape_raw/{subject}/scans_ply/{seq}/{frame}'
obj_path = f'./data/cape_raw/{subject}/scans/{seq}/{os.path.splitext(frame)[0]}.obj'

# Load the PLY file
mesh = pywavefront.Wavefront(ply_path, encoding='gb2312')

# Extract the vertices, texture coordinates, normals and colors from the mesh
vertices = mesh.vertices
texcoords = mesh.texcoords
normals = mesh.normals
colors = mesh.materials[mesh.material_name]['diffuse']


# Write the OBJ file
with open(obj_path, 'w') as obj_file:
    for i in range(len(vertices)):
        obj_file.write('v {} {} {}\n'.format(vertices[i][0], vertices[i][1], vertices[i][2]))
    for i in range(len(texcoords)):
        obj_file.write('vt {} {}\n'.format(texcoords[i][0], texcoords[i][1]))
    for i in range(len(normals)):
        obj_file.write('vn {} {} {}\n'.format(normals[i][0], normals[i][1], normals[i][2]))
    for i in range(len(vertices)):
        obj_file.write('vc {} {} {}\n'.format(colors[0], colors[1], colors[2]))
    for face in mesh.faces:
        obj_file.write('f ')
        for vertex in face.vertices:
            obj_file.write('{}/{} {}/{} {}/{} '.format(vertex[0], vertex[1], vertex[0], vertex[1], vertex[0], vertex[1]))
        obj_file.write('\n')