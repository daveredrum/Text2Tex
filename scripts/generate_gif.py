# common utils
import os
import json
import argparse

from tqdm import tqdm

# pytorch3d
from pytorch3d.io import (
    load_obj,
    load_objs_as_meshes,
    save_obj
)
from pytorch3d.renderer import (
    PerspectiveCameras,
    AmbientLights,
    RasterizationSettings,
    look_at_view_transform,
    MeshRendererWithFragments,
    MeshRasterizer,
    HardFlatShader,
    SoftPhongShader,
    TexturesUV
)

# torch
import torch

from torchvision import transforms

# GIF
import imageio.v2 as imageio

# Setup
if torch.cuda.is_available():
    DEVICE = torch.device("cuda:0")
    torch.cuda.set_device(DEVICE)
else:
    print("no gpu avaiable")
    exit()

IMAGE_SIZE = 768

def init_args():
    print("=> initializing input arguments...")
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--obj_path", type=str, required=True)
    parser.add_argument("--num_views", type=int, default=8)

    parser.add_argument("--force", action="store_true", help="forcefully generate more image")

    # camera parameters NOTE need careful tuning!!!
    parser.add_argument("--test_camera", action="store_true")
    parser.add_argument("--dist", type=float, default=1.5, 
        help="distance to the camera from the object")
    parser.add_argument("--elev", type=float, default=15,
        help="the angle between the vector from the object to the camera and the horizontal plane")
    parser.add_argument("--azim", type=float, default=180,
        help="the angle between the vector from the object to the camera and the vertical plane")

    args = parser.parse_args()

    return args


def init_mesh(args, device=DEVICE):
    model_path = args.obj_path
    
    verts, faces, aux = load_obj(model_path, device=device)
    mesh = load_objs_as_meshes([model_path], device=device)

    return mesh, verts, faces, aux


def apply_offsets_to_mesh(mesh, offsets):
    new_mesh = mesh.offset_verts(offsets)

    return new_mesh


def init_camera(args, view_idx, device=DEVICE):
    interval = 360 // args.num_views
    dist = args.dist
    elev = args.elev
    azim = (args.azim + interval * view_idx) % 360
    R, T = look_at_view_transform(dist, elev, azim)
    image_size = torch.tensor([IMAGE_SIZE, IMAGE_SIZE]).unsqueeze(0)
    cameras = PerspectiveCameras(R=R, T=T, device=device, image_size=image_size)

    return cameras, dist, elev, azim


def init_renderer(camera, device=DEVICE):
    raster_settings = RasterizationSettings(image_size=IMAGE_SIZE)
    lights = AmbientLights(device=device)
    renderer = MeshRendererWithFragments(
        rasterizer=MeshRasterizer(
            cameras=camera,
            raster_settings=raster_settings
        ),
        shader=SoftPhongShader(
            cameras=camera,
            lights=lights,
            device=device
        )
    )

    return renderer


def render(mesh, renderer):
    images, fragments = renderer(mesh)

    return images, fragments


def save_args(args, output_dir):
    with open(os.path.join(output_dir, "args.json"), "w") as f:
        json.dump(
            {k: v for k, v in vars(args).items()},
            f,
            indent=4
        )
        

if __name__ == "__main__":
    args = init_args()

    # save
    output_dir = os.path.join(args.input_dir, "GIF-{}".format(args.num_views))
    os.makedirs(output_dir, exist_ok=True)

    # init resources
    # init mesh
    mesh, verts, faces, aux = init_mesh(args)

    num_verts = mesh.verts_packed().shape[0]
    mesh_center = mesh.verts_packed().mean(dim=0, keepdim=True).repeat(num_verts, 1)
    mesh = apply_offsets_to_mesh(mesh, -mesh_center)

    # save args
    save_args(args, output_dir)

    # rendering
    print("=> rendering...")
    for view_idx in tqdm(range(args.num_views)):

        init_image_path = os.path.join(output_dir, "{}.png".format(view_idx))

        if not os.path.exists(init_image_path) or args.force:

            # render the view
            cameras, dist, elev, azim = init_camera(args, view_idx)
            renderer = init_renderer(cameras)
            init_images_tensor, fragments = render(mesh, renderer)

            # save images
            init_image = init_images_tensor[0].cpu()
            init_image = init_image.permute(2, 0, 1)
            init_image = transforms.ToPILImage()(init_image).convert("RGB")
            init_image.save(init_image_path)

    # generate GIF
    images = [imageio.imread(os.path.join(output_dir, "{}.png").format(v_id)) for v_id in range(args.num_views)]
    imageio.mimsave(os.path.join(output_dir, "output.gif"), images, duration=1)

    print("=> done!")
