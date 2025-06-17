import json
import os
import torch.nn as nn
import cv2
import numpy as np
import torch
from diffusers import (AutoencoderKL, CogVideoXDDIMScheduler, DDIMScheduler,
                       DPMSolverMultistepScheduler,
                       EulerAncestralDiscreteScheduler, EulerDiscreteScheduler,
                       PNDMScheduler)
from omegaconf import OmegaConf
from PIL import Image
from transformers import (CLIPImageProcessor, CLIPVisionModelWithProjection,
                          T5EncoderModel, T5Tokenizer, AutoModel)


from cogvideox.models.autoencoder_magvit import AutoencoderKLCogVideoX
from cogvideox.models.transformer3d_radio import CogVideoXTransformer3DModel
from cogvideox.models.transformer3d import CogVideoXTransformer3DModel as CogVideoXTransformer3DModel_ori

from cogvideox.utils.lora_utils import merge_lora, unmerge_lora
from cogvideox.utils.utils import get_ref_video_to_video_latent, save_videos_grid, save_com_gif
from cogvideox.pipeline.pipeline_cogvideo_color_ref import CogVideoX_Fun_Pipeline_Control_Color



# Low gpu memory mode, this is used when the GPU memory is under 16GB
low_gpu_memory_mode = False

# model path
model_name          = "alibaba-pai/CogVideoX-Fun-V1.1-2b-Pose"

# Choose the sampler in "Euler" "Euler A" "DPM++" "PNDM" and "DDIM"
sampler_name        = "DDIM_Origin"

# Load pretrained model if need
vae_path            = None
lora_path           = None
# Other params
# sample_size         = [480, 848]
video_length = 46 

fps                 = None

# Use torch.float16 if GPU does not support torch.bfloat16
# ome graphics cards, such as v100, 2080ti, do not support torch.bfloat16
weight_dtype            = torch.bfloat16



# prompt = "" #the video is an animated scene featuring two characters in a dynamic combat situation. the character in the foreground is a muscular figure with a dark complexion, wearing a black armor with gold accents and a helmet with a skull design. this character is wielding a large sword with a curved blade and a gold handle. the character in the background is a smaller figure with a lighter complexion, wearing a black outfit with a hood and a mask that covers the eyes. this character is holding a staff with a curved top. the background is a plain blue sky with no visible landmarks or text. the characters are in motion, suggesting a fast-paced action sequence. the style of the animation is reminiscent of japanese anime, with detailed character designs and fluid movement. there is no visible text in the video." #The video features an athlete engaged in a volleyball match. The athlete is dressed in a striking red and black uniform, contrasting vividly against the dynamic backdrop of the volleyball court. The volleyball itself is uniquely colored with alternating yellow and blue panels, making it a prominent visual element as it arcs through the air. The scene captures the intensity and agility of the athlete as they leap and spike the ball, highlighting the fluidity of motion and the vibrant colors of the game. As the play progresses, the athlete stretches out their hand to skillfully hit the volleyball, showcasing their precision and timing."
ref_image_path = "./example/reference/10.png"
control_video = "./example/sketch/10.mp4"
with open('./example/caption/10.txt', 'r', encoding='utf-8') as f:
    prompt = f.read().strip()
guidance_scale          = 6.0
seed                    = 43
num_inference_steps     = 50
lora_weight             = 0.55
save_path               = "./results/"
transformer_name        = "./checkpoints"
negative_prompt         = "The video is not of a high quality, it has a low resolution. Watermark present in each frame. The background is solid. Strange body and strange trajectory. Distortion. "



denoising_transformer = CogVideoXTransformer3DModel.from_pretrained(
    transformer_name,
    subfolder="transformer",
    torch_dtype=weight_dtype,
)

reference_transformer = CogVideoXTransformer3DModel_ori.from_pretrained(
    transformer_name,
    subfolder="referencenet",
    torch_dtype=weight_dtype,
)

# Get Vae
vae = AutoencoderKLCogVideoX.from_pretrained(
    model_name, 
    subfolder="vae"
).to(weight_dtype)

if vae_path is not None:
    print(f"From checkpoint: {vae_path}")
    if vae_path.endswith("safetensors"):
        from safetensors.torch import load_file, safe_open
        state_dict = load_file(vae_path)
    else:
        state_dict = torch.load(vae_path, map_location="cpu")
    state_dict = state_dict["state_dict"] if "state_dict" in state_dict else state_dict

    m, u = vae.load_state_dict(state_dict, strict=False)
    print(f"missing keys: {len(m)}, unexpected keys: {len(u)}")

text_encoder = T5EncoderModel.from_pretrained(
    model_name, subfolder="text_encoder", torch_dtype=weight_dtype
)
dclip_model = AutoModel.from_pretrained("nvidia/RADIO", trust_remote_code=True, torch_dtype=weight_dtype).to("cuda")
dclip_processor = CLIPImageProcessor.from_pretrained("nvidia/RADIO", torch_dtype=weight_dtype)
            
        
# Get Scheduler
Choosen_Scheduler = scheduler_dict = {
    "Euler": EulerDiscreteScheduler,
    "Euler A": EulerAncestralDiscreteScheduler,
    "DPM++": DPMSolverMultistepScheduler, 
    "PNDM": PNDMScheduler,
    "DDIM_Cog": CogVideoXDDIMScheduler,
    "DDIM_Origin": DDIMScheduler,
}[sampler_name]
scheduler = Choosen_Scheduler.from_pretrained(
    model_name, 
    subfolder="scheduler"
)

pipeline = CogVideoX_Fun_Pipeline_Control_Color.from_pretrained(
    model_name,
    vae=vae,
    text_encoder=text_encoder,
    denoising_transformer=denoising_transformer,
    reference_transformer=reference_transformer,
    scheduler=scheduler,
    torch_dtype=weight_dtype
)

if low_gpu_memory_mode:
    pipeline.enable_sequential_cpu_offload()
else:
    pipeline.enable_model_cpu_offload()

generator = torch.Generator(device="cuda").manual_seed(seed)

if lora_path is not None:
    pipeline = merge_lora(pipeline, lora_path, lora_weight, "cuda")

                

basename = os.path.basename(control_video)
ref_image, input_video, input_video_mask, _,_ = get_ref_video_to_video_latent(ref_image_path, control_video, video_length=video_length, fps=fps)
video_length = input_video.shape[2]
height = input_video.shape[3]#//16*16
width = input_video.shape[4]#//16*16
    
dclip_cond = dclip_processor(images=Image.open(ref_image_path).convert("RGB").resize([width, height]), return_tensors="pt").pixel_values.to("cuda", dtype=weight_dtype)
id_cond, id_vit_hidden = dclip_model(dclip_cond)

with torch.no_grad():
    sample = pipeline(   
        prompt, 
        num_frames = video_length,
        negative_prompt = negative_prompt,
        height      = height,
        width       = width,
        generator   = generator,
        guidance_scale = guidance_scale,
        num_inference_steps = num_inference_steps,
        ref_image = ref_image,
        control_video = input_video,
        id_cond = id_cond,
        id_vit_hidden=id_vit_hidden
    ).videos

if lora_path is not None:
    pipeline = unmerge_lora(pipeline, lora_path, lora_weight, "cuda")
    
if not os.path.exists(save_path):
    os.makedirs(save_path, exist_ok=True)


prefix = basename
    
if video_length == 1:
    save_sample_path = os.path.join(save_path, prefix)

    image = sample[0, :, 0]
    image = image.transpose(0, 1).transpose(1, 2)
    image = (image * 255).numpy().astype(np.uint8)
    image = Image.fromarray(image)
    image.save(save_sample_path)
else:
    video_path = os.path.join(save_path, prefix )
    save_videos_grid(sample, video_path, fps=24)
    