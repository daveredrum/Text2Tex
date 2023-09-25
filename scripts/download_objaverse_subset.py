import os
import sys
import objaverse
import subprocess
import xatlas
import trimesh

from PIL import Image
from pathlib import Path
from tqdm import tqdm

# torch
import torch

from torchvision import transforms

# pytorch3d
from pytorch3d.io import (
    load_obj,
    save_obj
)

# customized
sys.path.append(".")
from scripts.parameterize_mesh import parameterize_mesh

SUBSET = "./data/objaverse_subset.txt"
DATA_DIR = "./data/objaverse"

# Setup
if torch.cuda.is_available():
    DEVICE = torch.device("cuda:0")
    torch.cuda.set_device(DEVICE)
else:
    DEVICE = torch.device("cpu")

os.makedirs(DATA_DIR, exist_ok=True)

def get_objaverse_subset():
    with open(SUBSET) as f:
        ids = [l.rstrip().split("_")[-1] for l in f.readlines()]

    return ids

def adjust_uv_map(faces, aux, device=DEVICE):
    """
        adjust UV map to be compatiable with multiple textures.
        UVs for different materials will be decomposed and placed horizontally

        +-----+-----+-----+--
        |  1  |  2  |  3  |
        +-----+-----+-----+--

    """

    textures_ids = faces.textures_idx
    materials_idx = faces.materials_idx
    verts_uvs = aux.verts_uvs

    num_materials = torch.unique(materials_idx).shape[0]

    try:
        new_verts_uvs = verts_uvs.clone()

        # HACK map verts_uvs to 0 and 1
        new_verts_uvs[new_verts_uvs != 1] %= 1

        for material_id in range(num_materials):
            # apply offsets to horizontal axis
            faces_ids = textures_ids[materials_idx == material_id].unique()
            new_verts_uvs[faces_ids, 0] += material_id

        new_verts_uvs[:, 0] /= num_materials
        new_faces_uvs = faces.textures_idx
    except AttributeError:
        new_verts_uvs = torch.tensor([
            [0, 0],
            [0, 1],
            [1, 0]
        ]).to(device)
        
        num_faces = faces.verts_idx.shape[0]
        new_faces_uvs = torch.tensor([[0, 1, 2]]).to(device).long()
        new_faces_uvs = new_faces_uvs.repeat(num_faces, 1)

    return new_verts_uvs, new_faces_uvs

def load_and_adjust_mesh(mesh_path, device=DEVICE):
    verts, faces, aux = load_obj(mesh_path, device=device)

    dummy_texture = Image.open("./samples/textures/white.png").convert("RGB").resize((512, 512))

    # collapse textures of multiple materials to one texture map
    new_verts_uvs, new_faces_uvs = adjust_uv_map(faces, aux)

    return verts, faces, new_verts_uvs, new_faces_uvs, dummy_texture

def collapse_objects(input_path, output_path, device=DEVICE, inplace=False):
    verts, faces, new_verts_uvs, new_faces_uvs, dummy_texture = load_and_adjust_mesh(input_path)
    output_path = input_path if inplace else output_path
    os.makedirs(output_path.parent, exist_ok=True)

    texture_map = transforms.ToTensor()(dummy_texture).to(device)
    texture_map = texture_map.permute(1, 2, 0)

    save_obj(
        str(output_path),
        verts=verts,
        faces=faces.verts_idx,
        decimal_places=5,
        verts_uvs=new_verts_uvs,
        faces_uvs=new_faces_uvs,
        texture_map=texture_map
    )

def remove_tails(mtl_path):
    with open(mtl_path) as f:
        mtl_data = [l.rstrip() for l in f.readlines()]

    with open(mtl_path, "w") as f:
        for l in mtl_data:
            if "map_Bump" not in l and "map_Kd" not in l:
                f.write(l+'\n')

if __name__ == "__main__":
    objaverse_subset = get_objaverse_subset()

    # cache objects to ~/.objaverse/
    objects = objaverse.load_objects(objaverse_subset)

    print("=> processing...")
    for key, value in tqdm(objects.items()):
        # convert glb to obj
        cmd = [
            "blender", # NOTE please make sure you installed blender
            "--background", "--factory-startup", "--python", "scripts/blender_process_glb.py", "--",
            str(value),
            DATA_DIR
        ]
        _ = subprocess.call(cmd, stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)

        output_dir = Path(DATA_DIR) / str(Path(value).stem)

        obj_path = output_dir / "mesh.obj"
        mtl_path = output_dir / "mesh.mtl"

        remove_tails(mtl_path)
        collapse_objects(obj_path, obj_path, DEVICE, inplace=True)
        parameterize_mesh(obj_path, obj_path) # xatlas produces great UVs

    print("=> done!")