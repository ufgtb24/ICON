import os
from argparse import ArgumentParser

from plyfile import PlyData


def parse_args():
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

    return subject, seq,frame


def ply_path_to_obj_path(ply_path):
    """
    Replaces the .ply extension with .obj extension
    """
    return os.path.splitext(ply_path)[0] + '.obj'


def convert(ply_path, obj_path=None):
    """
    Converts the given .ply file to an .obj file
    """
    obj_path = obj_path or ply_path_to_obj_path(ply_path)
    ply = PlyData.read(ply_path)

    with open(obj_path, 'w') as f:
        f.write("# OBJ file\n")

        verteces = ply['vertex']

        for v in verteces:
            p = [v['x'], v['y'], v['z']]
            if 'red' in v and 'green' in v and 'blue' in v:
                c = [v['red'] / 256, v['green'] / 256, v['blue'] / 256]
            else:
                c = [0, 0, 0]
            a = p + c
            f.write("v %.6f %.6f %.6f %.6f %.6f %.6f \n" % tuple(a))

        for v in verteces:
            if 'nx' in v and 'ny' in v and 'nz' in v:
                n = (v['nx'], v['ny'], v['nz'])
                f.write("vn %.6f %.6f %.6f\n" % n)

        for v in verteces:
            if 's' in v and 't' in v:
                t = (v['s'], v['t'])
                f.write("vt %.6f %.6f\n" % t)

        if 'face' in ply:
            for i in ply['face']['vertex_indices']:
                f.write("f")
                for j in range(i.size):
                    # ii = [ i[j]+1 ]
                    ii = [i[j] + 1, i[j] + 1, i[j] + 1]
                    # f.write(" %d" % tuple(ii) )
                    f.write(" %d/%d/%d" % tuple(ii))
                f.write("\n")


def main():
    subject, seq,frame = parse_args()
    ply_path = f'./data/cape_raw/{subject}/scans_ply/{seq}/{frame}'
    obj_path = f'./data/cape_raw/{subject}/scans/{seq}/{os.path.splitext(frame)[0] }.obj'

    # obj_path = ply_path_to_obj_path(ply_path)
    print(f"Converting {ply_path} to .obj...")
    convert(ply_path, obj_path)
    print(f"Conversion finished successfully. Output path: {obj_path}")


if __name__ == '__main__':
    main()