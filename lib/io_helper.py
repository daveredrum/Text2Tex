# common utils
import os
import json

# numpy
import numpy as np

# visualization
import matplotlib
import matplotlib.cm as cm
import matplotlib.pyplot as plt

matplotlib.use("Agg")

from pytorch3d.io import save_obj

from torchvision import transforms


def save_depth(fragments, output_dir, init_image, view_idx):
    print("=> saving depth...")
    width, height = init_image.size
    dpi = 100
    figsize = width / float(dpi), height / float(dpi)

    depth_np = fragments.zbuf[0].cpu().numpy()

    fig = plt.figure(figsize=figsize)
    ax = fig.add_axes([0, 0, 1, 1])
    # Hide spines, ticks, etc.
    ax.axis('off')
    # Display the image.
    ax.imshow(depth_np, cmap='gray')

    plt.savefig(os.path.join(output_dir, "{}.png".format(view_idx)), bbox_inches='tight', pad_inches=0)
    np.save(os.path.join(output_dir, "{}.npy".format(view_idx)), depth_np[..., 0])


def save_backproject_obj(output_dir, obj_name, 
    verts, faces, verts_uvs, faces_uvs, projected_texture, 
    device):
    print("=> saving OBJ file...")
    texture_map = transforms.ToTensor()(projected_texture).to(device)
    texture_map = texture_map.permute(1, 2, 0)
    obj_path = os.path.join(output_dir, obj_name)

    save_obj(
        obj_path,
        verts=verts,
        faces=faces,
        decimal_places=5,
        verts_uvs=verts_uvs,
        faces_uvs=faces_uvs,
        texture_map=texture_map
    )


def save_args(args, output_dir):
    with open(os.path.join(output_dir, "args.json"), "w") as f:
        json.dump(
            {k: v for k, v in vars(args).items()},
            f,
            indent=4
        )


def save_viewpoints(args, output_dir, dist_list, elev_list, azim_list, view_list):
    with open(os.path.join(output_dir, "viewpoints.json"), "w") as f:
        json.dump(
            {
                "dist": dist_list,
                "elev": elev_list,
                "azim": azim_list,
                "view": view_list
            },
            f,
            indent=4
        )