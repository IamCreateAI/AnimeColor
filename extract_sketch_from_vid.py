import concurrent.futures
import os
import random
from pathlib import Path

import numpy as np
import sys
from PIL import Image
import numpy as np
import cv2
import importlib
import os
import os.path as osp
import shutil
import sys
from pathlib import Path

import av
import numpy as np
import torch
import torchvision
from einops import rearrange
from PIL import Image


def seed_everything(seed):
    import random

    import numpy as np

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed % (2**32))
    random.seed(seed)


def import_filename(filename):
    spec = importlib.util.spec_from_file_location("mymodule", filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def delete_additional_ckpt(base_path, num_keep):
    dirs = []
    for d in os.listdir(base_path):
        if d.startswith("checkpoint-"):
            dirs.append(d)
    num_tot = len(dirs)
    if num_tot <= num_keep:
        return
    # ensure ckpt is sorted and delete the ealier!
    del_dirs = sorted(dirs, key=lambda x: int(x.split("-")[-1]))[: num_tot - num_keep]
    for d in del_dirs:
        path_to_dir = osp.join(base_path, d)
        if osp.exists(path_to_dir):
            shutil.rmtree(path_to_dir)


def save_videos_from_pil(pil_images, path, fps=8):
    import av

    save_fmt = Path(path).suffix
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    try:
        width, height = pil_images[0].size
    except IndexError:
        print(f"Error: pil_images list is empty.")
        return

    if save_fmt == ".mp4":
        codec = "libx264"
        container = av.open(path, "w")
        stream = container.add_stream(codec, rate=fps)

        stream.width = width
        stream.height = height

        for pil_image in pil_images:
            # pil_image = Image.fromarray(image_arr).convert("RGB")
            av_frame = av.VideoFrame.from_image(pil_image)
            container.mux(stream.encode(av_frame))
        container.mux(stream.encode())
        container.close()

    elif save_fmt == ".gif":
        pil_images[0].save(
            fp=path,
            format="GIF",
            append_images=pil_images[1:],
            save_all=True,
            duration=(1 / fps * 1000),
            loop=0,
        )
    else:
        raise ValueError("Unsupported file type. Use .mp4 or .gif.")


def save_videos_grid(videos: torch.Tensor, path: str, rescale=False, n_rows=6, fps=8):
    videos = rearrange(videos, "b c t h w -> t b c h w")
    height, width = videos.shape[-2:]
    outputs = []

    for x in videos:
        x = torchvision.utils.make_grid(x, nrow=n_rows)  # (c h w)
        x = x.transpose(0, 1).transpose(1, 2).squeeze(-1)  # (h w c)
        if rescale:
            x = (x + 1.0) / 2.0  # -1,1 -> 0,1
        x = (x * 255).numpy().astype(np.uint8)
        x = Image.fromarray(x)

        outputs.append(x)

    os.makedirs(os.path.dirname(path), exist_ok=True)

    save_videos_from_pil(outputs, path, fps)


def read_frames(video_path):
    try:
        container = av.open(video_path)
        video_stream = next(s for s in container.streams if s.type == "video")
        frames = []
        for packet in container.demux(video_stream):
            for frame in packet.decode():
                image = Image.frombytes(
                    "RGB",
                    (frame.width, frame.height),
                    frame.to_rgb().to_ndarray(),
                )
                frames.append(image)
        return frames
    except UnicodeDecodeError:
        print(f"UnicodeDecodeError: unable to open video at {video_path} due to invalid characters in path or content.")
        return []


def get_fps(video_path):
    try:
        container = av.open(video_path)
        video_stream = next(s for s in container.streams if s.type == "video")
        fps = video_stream.average_rate
        container.close()
        return fps
    except UnicodeDecodeError:
        print(f"UnicodeDecodeError: unable to open video at {video_path} due to invalid characters in path or content.")
        return None

def detector(image,
                kernel_size=0,
                sigma=1,
                k_sigma=1.6,
                epsilon=-1.3,
                phi=50,
                gamma=0.98):
    """XDoG(Extended Difference of Gaussians)を処理した画像を返す

    Args:
        image: OpenCV Image
        kernel_size: Gaussian Blur Kernel Size
        sigma: sigma for small Gaussian filter
        k_sigma: large/small for sigma Gaussian filter
        eps: threshold value between dark and bright
        phi: soft threshold
        gamma: scale parameter for DoG signal to make sharp

    Returns:
        Image after applying the XDoG.
    """
    epsilon /= 255
    dog = DoG_filter(image, kernel_size, sigma, k_sigma, gamma)
    dog /= dog.max()
    e = 1 + np.tanh(phi * (dog - epsilon))
    e[e >= 1] = 1
    return e.astype('uint8') * 255


def DoG_filter(image, kernel_size=0, sigma=1.4, k_sigma=1.6, gamma=1):
    """DoG(Difference of Gaussians)を処理した画像を返す

    Args:
        image: OpenCV Image
        kernel_size: Gaussian Blur Kernel Size
        sigma: sigma for small Gaussian filter
        k_sigma: large/small for sigma Gaussian filter
        gamma: scale parameter for DoG signal to make sharp

    Returns:
        Image after applying the DoG.
    """
    g1 = cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)
    g2 = cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma * k_sigma)
    return g1 - gamma * g2
# Extract dwpose mp4 videos from raw videos
# /path/to/video_dataset/*/*.mp4 -> /path/to/video_dataset_dwpose/*/*.mp4



def process_single_video(video_path, detector, root_dir, save_dir):
    relative_path = os.path.relpath(video_path, root_dir)
    print(relative_path, video_path, root_dir)
    out_path = os.path.join(save_dir, relative_path)
    if os.path.exists(out_path):
        return

    output_dir = Path(os.path.dirname(os.path.join(save_dir, relative_path)))
    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)

    fps = get_fps(video_path)
    frames = read_frames(video_path)
    kps_results = []
    for i, frame_pil in enumerate(frames):
        frame_pil = cv2.cvtColor(np.array(frame_pil), cv2.COLOR_RGB2BGR)
        gray_image = cv2.cvtColor(frame_pil, cv2.COLOR_BGR2GRAY)

        result = detector(gray_image)
        result = Image.fromarray(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))

        kps_results.append(result)

    save_videos_from_pil(kps_results, out_path, fps=fps)


def process_batch_videos(video_list, detector, root_dir, save_dir):
    for i, video_path in enumerate(video_list):
        print(f"Process {i}/{len(video_list)} video")
        process_single_video(video_path, detector, root_dir, save_dir)


if __name__ == "__main__":
    # -----
    # NOTE:
    # python tools/extract_dwpose_from_vid.py --video_root /path/to/video_dir
    # -----
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--video_root", type=str)
    parser.add_argument(
        "--save_dir", type=str, help="Path to save extracted pose videos"
    )
    parser.add_argument("-j", type=int, default=4, help="Num workers")
    args = parser.parse_args()
    num_workers = args.j
    if args.save_dir is None:
        save_dir = args.video_root + "_sketch"
    else:
        save_dir = args.save_dir
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # collect all video_folder paths
    video_mp4_paths = set()
    for root, dirs, files in os.walk(args.video_root):
        for name in files:
            if name.endswith(".mp4"):
                video_mp4_paths.add(os.path.join(root, name))
    video_mp4_paths = list(video_mp4_paths)
    random.shuffle(video_mp4_paths)

    # split into chunks,
    batch_size = (len(video_mp4_paths) + num_workers - 1) // num_workers
    print(f"Num videos: {len(video_mp4_paths)} {batch_size = }")
    video_chunks = [
        video_mp4_paths[i : i + batch_size]
        for i in range(0, len(video_mp4_paths), batch_size)
    ]

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = []
        for i, chunk in enumerate(video_chunks):

            futures.append(
                executor.submit(
                    process_batch_videos, chunk, detector, args.video_root, save_dir
                )
            )
        for future in concurrent.futures.as_completed(futures):
            future.result()
