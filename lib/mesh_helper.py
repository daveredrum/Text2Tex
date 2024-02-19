import os
import torch
import trimesh
import xatlas

import numpy as np

from sklearn.decomposition import PCA

from torchvision import transforms

from tqdm import tqdm

from pytorch3d.io import (
    load_obj,
    load_objs_as_meshes
)


def compute_principle_directions(model_path, num_points=20000):
    mesh = trimesh.load_mesh(model_path, force="mesh")
    pc, _ = trimesh.sample.sample_surface_even(mesh, num_points)

    pc -= np.mean(pc, axis=0, keepdims=True)

    principle_directions = PCA(n_components=3).fit(pc).components_
    
    return principle_directions


def init_mesh(input_path, cache_path, device):
    print("=> parameterizing target mesh...")

    mesh = trimesh.load_mesh(input_path, force='mesh')
    try:
        vertices, faces = mesh.vertices, mesh.faces
    except AttributeError:
        print("multiple materials in {} are not supported".format(input_path))
        exit()

    vmapping, indices, uvs = xatlas.parametrize(vertices, faces)
    xatlas.export(str(cache_path), vertices[vmapping], indices, uvs)

    print("=> loading target mesh...")

    # principle_directions = compute_principle_directions(cache_path)
    principle_directions = None
    
    _, faces, aux = load_obj(cache_path, device=device)
    mesh = load_objs_as_meshes([cache_path], device=device)

    num_verts = mesh.verts_packed().shape[0]

    # make sure mesh center is at origin
    bbox = mesh.get_bounding_boxes()
    mesh_center = bbox.mean(dim=2).repeat(num_verts, 1)
    mesh = apply_offsets_to_mesh(mesh, -mesh_center)

    # make sure mesh size is normalized
    box_size = bbox[..., 1] - bbox[..., 0]
    box_max = box_size.max(dim=1, keepdim=True)[0].repeat(num_verts, 3)
    mesh = apply_scale_to_mesh(mesh, 1 / box_max)

    return mesh, mesh.verts_packed(), faces, aux, principle_directions, mesh_center, box_max


def apply_offsets_to_mesh(mesh, offsets):
    new_mesh = mesh.offset_verts(offsets)

    return new_mesh

def apply_scale_to_mesh(mesh, scale):
    new_mesh = mesh.scale_verts(scale)

    return new_mesh


def adjust_uv_map(faces, aux, init_texture, uv_size):
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

    new_verts_uvs = verts_uvs.clone()
    for material_id in range(num_materials):
        # apply offsets to horizontal axis
        faces_ids = textures_ids[materials_idx == material_id].unique()
        new_verts_uvs[faces_ids, 0] += material_id

    new_verts_uvs[:, 0] /= num_materials

    init_texture_tensor = transforms.ToTensor()(init_texture)
    init_texture_tensor = torch.cat([init_texture_tensor for _ in range(num_materials)], dim=-1)
    init_texture = transforms.ToPILImage()(init_texture_tensor).resize((uv_size, uv_size))

    return new_verts_uvs, init_texture


@torch.no_grad()
def update_face_angles(mesh, cameras, fragments):
    def get_angle(x, y):
        x = torch.nn.functional.normalize(x)
        y = torch.nn.functional.normalize(y)
        inner_product = (x * y).sum(dim=1)
        x_norm = x.pow(2).sum(dim=1).pow(0.5)
        y_norm = y.pow(2).sum(dim=1).pow(0.5)
        cos = inner_product / (x_norm * y_norm)
        angle = torch.acos(cos)
        angle = angle * 180 / 3.14159

        return angle

    # face normals
    face_normals = mesh.faces_normals_padded()[0]

    # view vector (object center -> camera center)
    camera_center = cameras.get_camera_center()

    face_angles = get_angle(
        face_normals, 
        camera_center.repeat(face_normals.shape[0], 1)
    ) # (F)

    face_angles_rev = get_angle(
        face_normals, 
        -camera_center.repeat(face_normals.shape[0], 1)
    ) # (F)

    face_angles = torch.minimum(face_angles, face_angles_rev)

    # Indices of unique visible faces
    visible_map = fragments.pix_to_face.unique()  # (num_visible_faces)
    invisible_mask = torch.ones_like(face_angles)
    invisible_mask[visible_map] = 0
    face_angles[invisible_mask == 1] = 10000.  # angles of invisible faces are ignored

    return face_angles
