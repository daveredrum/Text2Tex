import os
import argparse
import torch

import numpy as np

from pathlib import Path
from PIL import Image
from pytorch3d.io import (
    load_obj,
    save_obj
)


def init_args():
    print("=> initializing input arguments...")
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--obj_name", type=str, required=True)

    # rotation
    parser.add_argument("--plane", type=str, choices=["xy", "xz", "yz"], required=True)
    parser.add_argument("--deg", type=float, required=True, help="rotation degree")

    args = parser.parse_args()

    return args

def init_mesh(args):
    print("=> loading target mesh...")
    model_path = os.path.join(args.input_dir, "{}.obj".format(args.obj_name))
    
    verts, faces, aux = load_obj(model_path)

    texture_map = Image.open(str(Path(args.input_dir) / args.obj_name) + ".png").convert("RGB")
    texture_map = torch.from_numpy(np.array(texture_map)) / 255.

    return verts, faces, aux, texture_map

def rotate_verts(args, verts):
    theta = args.deg
    theta = theta * np.pi / 180
    A = torch.tensor([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta), np.cos(theta)]
    ]).float()

    if args.plane == "xy":
        xy = torch.stack([verts[:, 0], verts[:, 1]], dim=1).float()
        xy = torch.matmul(A, xy.T).T
        verts = torch.stack([xy[:, 0], xy[:, 1], verts[:, 2]], dim=1)
    elif args.plane == "xz":
        xz = torch.stack([verts[:, 0], verts[:, 2]], dim=1).float()
        xz = torch.matmul(A, xz.T).T
        verts = torch.stack([xz[:, 0], verts[:, 1], xz[:, 1]], dim=1)
    else:
        yz = torch.stack([verts[:, 1], verts[:, 2]], dim=1).float()
        yz = torch.matmul(A, yz.T).T

        verts = torch.stack([verts[:, 0], yz[:, 0], yz[:, 1]], dim=1)

    return verts

if __name__ == "__main__":
    args = init_args()

    verts, faces, aux, texture_map = init_mesh(args)
    verts = rotate_verts(args, verts)

    save_obj(
        str(Path(args.input_dir) / args.obj_name) + "_rotated.obj",
        verts=verts,
        faces=faces.verts_idx,
        decimal_places=5,
        verts_uvs=aux.verts_uvs,
        faces_uvs=faces.textures_idx,
        texture_map=texture_map
    )

