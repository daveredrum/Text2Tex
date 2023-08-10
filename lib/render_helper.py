import os
import torch

import cv2

import numpy as np

from PIL import Image

from torchvision import transforms
from pytorch3d.ops import interpolate_face_attributes
from pytorch3d.renderer import (
    RasterizationSettings,
    MeshRendererWithFragments,
    MeshRasterizer,
)

# customized
import sys
sys.path.append(".")


def init_renderer(camera, shader, image_size, faces_per_pixel):
    raster_settings = RasterizationSettings(image_size=image_size, faces_per_pixel=faces_per_pixel)
    renderer = MeshRendererWithFragments(
        rasterizer=MeshRasterizer(
            cameras=camera,
            raster_settings=raster_settings
        ),
        shader=shader
    )

    return renderer


@torch.no_grad()
def render(mesh, renderer, pad_value=10):
    def phong_normal_shading(meshes, fragments) -> torch.Tensor:
        faces = meshes.faces_packed()  # (F, 3)
        vertex_normals = meshes.verts_normals_packed()  # (V, 3)
        faces_normals = vertex_normals[faces]
        pixel_normals = interpolate_face_attributes(
            fragments.pix_to_face, fragments.bary_coords, faces_normals
        )

        return pixel_normals

    def similarity_shading(meshes, fragments):
        faces = meshes.faces_packed()  # (F, 3)
        vertex_normals = meshes.verts_normals_packed()  # (V, 3)
        faces_normals = vertex_normals[faces]
        vertices = meshes.verts_packed()  # (V, 3)
        face_positions = vertices[faces]
        view_directions = torch.nn.functional.normalize((renderer.shader.cameras.get_camera_center().reshape(1, 1, 3) - face_positions), p=2, dim=2)
        cosine_similarity = torch.nn.CosineSimilarity(dim=2)(faces_normals, view_directions)
        pixel_similarity = interpolate_face_attributes(
            fragments.pix_to_face, fragments.bary_coords, cosine_similarity.unsqueeze(-1)
        )

        return pixel_similarity

    def get_relative_depth_map(fragments, pad_value=pad_value):
        absolute_depth = fragments.zbuf[..., 0] # B, H, W
        no_depth = -1

        depth_min, depth_max = absolute_depth[absolute_depth != no_depth].min(), absolute_depth[absolute_depth != no_depth].max()
        target_min, target_max = 50, 255

        depth_value = absolute_depth[absolute_depth != no_depth]
        depth_value = depth_max - depth_value # reverse values

        depth_value /= (depth_max - depth_min)
        depth_value = depth_value * (target_max - target_min) + target_min

        relative_depth = absolute_depth.clone()
        relative_depth[absolute_depth != no_depth] = depth_value
        relative_depth[absolute_depth == no_depth] = pad_value # not completely black

        return relative_depth


    images, fragments = renderer(mesh)
    normal_maps = phong_normal_shading(mesh, fragments).squeeze(-2)
    similarity_maps = similarity_shading(mesh, fragments).squeeze(-2) # -1 - 1
    depth_maps = get_relative_depth_map(fragments)

    # normalize similarity mask to 0 - 1 
    similarity_maps = torch.abs(similarity_maps) # 0 - 1
    
    # HACK erode, eliminate isolated dots
    non_zero_similarity = (similarity_maps > 0).float()
    non_zero_similarity = (non_zero_similarity * 255.).cpu().numpy().astype(np.uint8)[0]
    non_zero_similarity = cv2.erode(non_zero_similarity, kernel=np.ones((3, 3), np.uint8), iterations=2)
    non_zero_similarity = torch.from_numpy(non_zero_similarity).to(similarity_maps.device).unsqueeze(0) / 255.
    similarity_maps = non_zero_similarity.unsqueeze(-1) * similarity_maps 

    return images, normal_maps, similarity_maps, depth_maps, fragments


@torch.no_grad()
def check_visible_faces(mesh, fragments):
    pix_to_face = fragments.pix_to_face

    # Indices of unique visible faces
    visible_map = pix_to_face.unique()  # (num_visible_faces)

    return visible_map

