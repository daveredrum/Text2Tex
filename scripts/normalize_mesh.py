# common utils
import os
import argparse

# torch
import torch

# pytorch3d
from pytorch3d.io import (
    load_obj,
    load_objs_as_meshes,
    save_obj
)

# Setup
if torch.cuda.is_available():
    DEVICE = torch.device("cuda:0")
    torch.cuda.set_device(DEVICE)
else:
    DEVICE = torch.device("cpu")

def init_args():
    print("=> initializing input arguments...")
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--obj_name", type=str, required=True)

    args = parser.parse_args()

    return args


def init_mesh(args, device=DEVICE):
    print("=> loading target mesh...")
    model_path = os.path.join(args.input_dir, "{}.obj".format(args.obj_name))
    
    verts, faces, aux = load_obj(model_path, device=device)
    mesh = load_objs_as_meshes([model_path], device=device)

    return mesh, verts, faces, aux
    

def normalize_mesh(mesh):
    bbox = mesh.get_bounding_boxes()
    num_verts = mesh.verts_packed().shape[0]

    # move mesh to origin
    mesh_center = bbox.mean(dim=2).repeat(num_verts, 1)
    mesh = mesh.offset_verts(-mesh_center)

    # scale
    lens = bbox[0, :, 1] - bbox[0, :, 0]
    max_len = lens.max()
    scale = 1 / max_len
    scale = scale.unsqueeze(0).repeat(num_verts)
    mesh.scale_verts_(scale)

    return mesh.verts_packed()


def save_normalized_obj(args, verts, faces, aux):
    print("=> saving backprojected OBJ file...")
    obj_path = os.path.join(args.input_dir, "{}_normalized.obj".format(args.obj_name))

    save_obj(
        obj_path,
        verts=verts,
        faces=faces.verts_idx,
        decimal_places=5,
        verts_uvs=aux.verts_uvs,
        faces_uvs=faces.textures_idx,
        texture_map=aux.texture_images[list(aux.texture_images.keys())[0]]
    )


if __name__ == "__main__":
    args = init_args()

    mesh, verts, faces, aux = init_mesh(args)
    verts = normalize_mesh(mesh)

    save_normalized_obj(args, verts, faces, aux)
