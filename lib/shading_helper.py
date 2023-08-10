from typing import NamedTuple, Sequence

from pytorch3d.renderer.mesh.shader import ShaderBase
from pytorch3d.renderer import (
    AmbientLights,
    SoftPhongShader
)


class BlendParams(NamedTuple):
    sigma: float = 1e-4
    gamma: float = 1e-4
    background_color: Sequence = (1, 1, 1)


class FlatTexelShader(ShaderBase):

    def __init__(self, device="cpu", cameras=None, lights=None, materials=None, blend_params=None):
        super().__init__(device, cameras, lights, materials, blend_params)

    def forward(self, fragments, meshes, **_kwargs):
        texels = meshes.sample_textures(fragments)
        texels[(fragments.pix_to_face == -1), :] = 0
        return texels.squeeze(-2)


def init_soft_phong_shader(camera, blend_params, device):
    lights = AmbientLights(device=device)
    shader = SoftPhongShader(
        cameras=camera,
        lights=lights,
        device=device,
        blend_params=blend_params
    )

    return shader


def init_flat_texel_shader(camera, device):
    shader=FlatTexelShader(
        cameras=camera,
        device=device
    )
    
    return shader