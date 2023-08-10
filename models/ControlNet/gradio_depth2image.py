import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

from share import *
import config

import cv2
import einops
import gradio as gr
import numpy as np
import torch
import random

from PIL import Image
from torchvision import transforms

from pytorch_lightning import seed_everything
from annotator.util import resize_image, HWC3
from cldm.model import create_model, load_state_dict
from models.ControlNet.ldm.models.diffusion.ddim import DDIMSampler


def init_model():
    model = create_model(BASE_DIR+'/models/cldm_v15.yaml').cpu()
    state_dict = load_state_dict(BASE_DIR+'/models/control_sd15_depth.pth')
    model.load_state_dict(state_dict, strict=False)
    # model.load_state_dict(state_dict)
    model = model.cuda()
    ddim_sampler = DDIMSampler(model)

    return model, ddim_sampler


@torch.no_grad()
def process(model, ddim_sampler, input_image, prompt, a_prompt, n_prompt, num_samples, 
    ddim_steps, scale, seed, eta, 
    strength=1.0, detected_map=None, unknown_mask=None, save_memory=False, depth_pad=10):

    """
        unknown mask has to be an array of shape (H, W) - should has values of (0, 255)
    """
    
    with torch.no_grad():
        H, W, C = input_image.shape

        if seed == -1:
            seed = random.randint(0, 65535)
        seed_everything(seed)

        if save_memory:
            model.low_vram_shift(is_diffusing=False)

        

        if save_memory:
            model.low_vram_shift(is_diffusing=True)

        # start from noising the input image
        x0 = Image.fromarray(input_image).convert("RGB")
        x0 = np.array(x0)

        x0 = torch.from_numpy(x0).permute(2, 0, 1).float().to(model.device)
        x0 = x0.unsqueeze(0).repeat(num_samples, 1, 1, 1)
        x0 = (x0 / 127.5) - 1.0 # NOTE input image must be normalized to [-1, 1]

        # encode input image
        # NOTE ControlNet doesn't accept the raw input image
        x0 = model.encode_first_stage(x0)
        x0 = model.get_first_stage_encoding(x0).detach()

        ddim_sampler.make_schedule(ddim_num_steps=ddim_steps, ddim_eta=eta, verbose=False, strength=strength)
        ddim_steps = int(ddim_steps * strength) # actually DEPRECATED

        # add noises to the maximum
        ddim_steps_tensor = torch.full((x0.shape[0],), ddim_sampler.ddim_timesteps[-1]).to(model.device)
        x_T = model.q_sample(x0, ddim_steps_tensor)

        # control
        if detected_map is None:
            detected_map, _ = apply_midas(resize_image(input_image, H))
            detected_map = HWC3(detected_map)

        detected_map_resized = cv2.resize(detected_map, (W, H), interpolation=cv2.INTER_LINEAR)

        control = torch.from_numpy(detected_map_resized).float().cuda() / 255.0
        control = torch.stack([control for _ in range(num_samples)], dim=0)
        control = einops.rearrange(control, 'b h w c -> b c h w').clone()

        cond = {"c_concat": [control], "c_crossattn": [model.get_learned_conditioning([prompt + ', ' + a_prompt] * num_samples)]}
        un_cond = {"c_concat": [control], "c_crossattn": [model.get_learned_conditioning([n_prompt] * num_samples)]}
        shape = (4, H // 8, W // 8)

        # if unknown_mask is not None:
        #     # HACK
        #     # unknown_mask_dilate = cv2.dilate(unknown_mask, kernel=np.ones((5, 5), np.uint8), iterations=2)
        #     unknown_mask_dilate = cv2.dilate(unknown_mask, kernel=np.ones((5, 5), np.uint8), iterations=1)

        #     # depth_mask = np.zeros_like(detected_map[..., 0])
        #     # depth_mask[detected_map[..., 0] != 0] = 1 # 1 -> object, 0 -> background
        #     # reversed_depth_mask = 1 - depth_mask # 0 -> object, 1 -> background

        #     # diffusion_mask = reversed_depth_mask + unknown_mask
        #     # unknown_mask_dilate = cv2.dilate(diffusion_mask, kernel=np.ones((5, 5), np.uint8), iterations=2)

        #     unknown_mask_dilate = Image.fromarray(unknown_mask_dilate.astype(np.uint8)).convert("L")
        #     unknown_mask_dilate = unknown_mask_dilate.resize((H // 8, W // 8), Image.NEAREST)
        #     unknown_mask_dilate = transforms.ToTensor()(unknown_mask_dilate).to(model.device)
        #     unknown_mask_dilate = unknown_mask_dilate.repeat(4, 1, 1)

        #     # only contains 0 and 1
        #     assert set(torch.unique(unknown_mask_dilate).cpu().numpy().tolist()).issubset(set([0, 1]))

        # else:
        #     unknown_mask_dilate = None

    
        if unknown_mask is not None:

            # # target: unknown region
            # unknown_mask_image = np.copy(unknown_mask) # should be 0 - 255
            # unknown_mask = unknown_mask.astype(np.float32)
            # unknown_mask /= 255 # normalize it to 0 - 1

            # target: unknown region + background
            # HACK basically generate everything except known region
            detected_map_image = Image.fromarray(detected_map.astype(np.uint8)).convert("L")
            detected_map_np = np.array(detected_map_image)
            background_mask = detected_map_np == depth_pad # bool
            background_mask = background_mask.astype(np.float32) * 255 # 0 - 255
            unknown_mask_image = unknown_mask + background_mask

            # unknown_mask is still the unknown region
            # will be used later to compose the generated region
            unknown_mask = unknown_mask.astype(np.float32)
            unknown_mask /= 255 # normalize it to 0 - 1

            compose_flag = True

        else:

            detected_map_image = Image.fromarray(detected_map.astype(np.uint8)).convert("L")
            detected_map_np = np.array(detected_map_image)

            # # target: non-background region
            # unknown_mask = (detected_map_np != depth_pad).astype(np.uint8)
            # unknown_mask_image = (unknown_mask * 255.).astype(np.uint8)
            # # Image.fromarray(unknown_mask_image).save("unknown.png")

            # target: everything
            unknown_mask = np.ones_like(detected_map_np)
            unknown_mask_image = (unknown_mask * 255.).astype(np.uint8)

            compose_flag = False


        # HACK
        # unknown_mask_dilate = np.copy(unknown_mask_image)
        unknown_mask_dilate = cv2.dilate(unknown_mask_image, kernel=np.ones((5, 5), np.uint8), iterations=2)
        unknown_mask_dilate = Image.fromarray(unknown_mask_dilate.astype(np.uint8)).convert("L")
        unknown_mask_dilate = unknown_mask_dilate.resize((H // 8, W // 8), Image.NEAREST)
        unknown_mask_dilate = transforms.ToTensor()(unknown_mask_dilate).to(model.device)
        unknown_mask_dilate = unknown_mask_dilate.repeat(4, 1, 1)

        # HACK make sure the mask only contains 0 and 1
        try:
            assert set(torch.unique(unknown_mask_dilate).cpu().numpy().tolist()).issubset(set([0, 1])), set(torch.unique(unknown_mask_dilate).cpu().numpy().tolist())
        except AssertionError:
            unknown_mask_dilate = torch.round(unknown_mask_dilate)

            assert set(torch.unique(unknown_mask_dilate).cpu().numpy().tolist()).issubset(set([0, 1])), set(torch.unique(unknown_mask_dilate).cpu().numpy().tolist())

        samples, intermediates = ddim_sampler.sample(
            ddim_steps, num_samples,
            shape, cond, x0=x0, x_T=x_T, mask=unknown_mask_dilate,
            verbose=False, eta=eta,
            unconditional_guidance_scale=scale,
            unconditional_conditioning=un_cond
        )

        if save_memory:
            model.low_vram_shift(is_diffusing=False)

        x_samples = model.decode_first_stage(samples)
        x_samples = (einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)

        results = []
        for i in range(num_samples):
            sample = x_samples[i]

            # # HACK manually compose cropped generated region with the known region
            # if compose_flag:
            #     # # unknown region + known region
            #     # input_image = Image.fromarray(input_image).convert("RGB")
            #     # input_image = np.array(input_image)

            #     # mask = np.repeat(unknown_mask[..., None], 3, axis=2)

            #     # new_sample = np.zeros_like(sample)
            #     # new_sample[mask == 1] = sample[mask == 1]
            #     # new_sample[mask == 0] = input_image[mask == 0]

            #     # sample = new_sample

            #     # non-background region
            #     detected_map_image = Image.fromarray(detected_map.astype(np.uint8)).convert("L")
            #     detected_map_np = np.array(detected_map_image)

            #     background_mask = (detected_map_np == depth_pad).astype(np.uint8)
            #     sample[background_mask == 1] = 255

            results.append(sample)

    return results


if __name__ == "__main__":
    model, ddim_sampler = init_model()

    block = gr.Blocks().queue()
    with block:
        with gr.Row():
            gr.Markdown("## Control Stable Diffusion with Depth Maps")
        with gr.Row():
            with gr.Column():
                input_image = gr.Image(source='upload', type="numpy")
                prompt = gr.Textbox(label="Prompt")
                run_button = gr.Button(label="Run")
                with gr.Accordion("Advanced options", open=False):
                    num_samples = gr.Slider(label="Images", minimum=1, maximum=12, value=1, step=1)
                    image_resolution = gr.Slider(label="Image Resolution", minimum=256, maximum=768, value=512, step=256)
                    detect_resolution = gr.Slider(label="Depth Resolution", minimum=128, maximum=1024, value=384, step=1)
                    ddim_steps = gr.Slider(label="Steps", minimum=1, maximum=100, value=20, step=1)
                    scale = gr.Slider(label="Guidance Scale", minimum=0.1, maximum=30.0, value=9.0, step=0.1)
                    seed = gr.Slider(label="Seed", minimum=-1, maximum=2147483647, step=1, randomize=True)
                    eta = gr.Number(label="eta (DDIM)", value=0.0)
                    a_prompt = gr.Textbox(label="Added Prompt", value='best quality, extremely detailed')
                    n_prompt = gr.Textbox(label="Negative Prompt",
                                        value='longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality')
            with gr.Column():
                result_gallery = gr.Gallery(label='Output', show_label=False, elem_id="gallery").style(grid=2, height='auto')
        ips = [model, ddim_sampler, input_image, prompt, a_prompt, n_prompt, num_samples, image_resolution, detect_resolution, ddim_steps, scale, seed, eta]
        run_button.click(fn=process, inputs=ips, outputs=[result_gallery])


    block.launch(server_name='0.0.0.0')
