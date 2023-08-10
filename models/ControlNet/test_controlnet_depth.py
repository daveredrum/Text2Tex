import os
import torch
import argparse

import cv2
from PIL import Image

import numpy as np

from gradio_depth2image import init_model, process


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_image", type=str, required=True)
    parser.add_argument("--input_depth", type=str, required=True)
    parser.add_argument("--prompt", type=str, required=True)
    
    # defaults
    parser.add_argument("--a_prompt", type=str, default="best quality, extremely detailed")
    parser.add_argument("--n_prompt", type=str, default="longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality")
    parser.add_argument("--num_samples", type=int, default=1)
    parser.add_argument("--image_resolution", type=int, default=768)
    parser.add_argument("--detect_resolution", type=int, default=768)
    parser.add_argument("--ddim_steps", type=int, default=20)
    parser.add_argument("--scale", type=float, default=7.5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eta", type=float, default=0.0)

    args = parser.parse_args()

    model, ddim_sampler = init_model()

    input_image = cv2.imread(args.input_image)
    depth_image = cv2.imread(args.input_depth)

    depth_min, depth_max = depth_image[depth_image != 0].min(), depth_image[depth_image != 0].max()
    depth_pad = 10
    assert depth_pad < depth_min

    depth_value = depth_image[depth_image != 0].astype(np.float32)
    depth_value = depth_max - depth_value

    depth_value /= (depth_max - depth_min)
    depth_value = depth_value * (depth_max - depth_min) + depth_min

    depth_image[depth_image != 0] = depth_value.astype(np.uint8)
    depth_image[depth_image == 0] = depth_pad # not completely black

    # depth_image = None

    outputs = process(
        model, ddim_sampler, 
        input_image, args.prompt, args.a_prompt, args.n_prompt, 
        args.num_samples, args.image_resolution, args.detect_resolution, 
        args.ddim_steps, args.scale, args.seed, args.eta, depth_image
    )

    for i in range(args.num_samples):
        out = outputs[i]
        Image.fromarray(out).save("control_depth_{}.png".format(i))
        
        out[depth_image == depth_pad] = 255 # crop output
        Image.fromarray(out).save("control_depth_{}_cropped.png".format(i))
