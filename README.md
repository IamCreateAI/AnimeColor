## AnimeColor: AnimeColor: Reference-based Animation Colorization with Diffusion Transformers

Official implementation of [**AnimeColor: AnimeColor: Reference-based Animation Colorization with Diffusion Transformers**](https://arxiv.org/abs/2507.20158), ACM MM 2025 

<div align="center"> <img src='assets/demo_s.gif'></img></div>


### Environment
```
conda create -n animecolor python=3.10
conda activate animecolor
pip install -r requirements.txt
```
### Checkpoints
please download the pre-trained animecolor checkpoints from [here](https://huggingface.co/rainbowow/AnimeColor) and put it into ./checkpoints

### Colorization
```
python test_msketch.py
```
Modify the settings of Line 52-61 to suit your needs:
```
ref_image_path = "./example/reference/1.png"
control_video = "./example/sketch/1.mp4"
with open('./example/caption/1.txt', 'r', encoding='utf-8') as f:
    prompt = f.read().strip()
guidance_scale          = 6.0
seed                    = 43
num_inference_steps     = 50
lora_weight             = 0.55
save_path               = "./results/"
transformer_name        = "./checkpoints"
```

You can also use the following script to extract sketches.
```
python extract_sketch_from_vid.py --video_root inputdir --save_dir outputdir
```
You can also use other sketch extraction methods, such like [AniLines-Anime-Lineart-Extractor](https://github.com/zhenglinpan/AniLines-Anime-Lineart-Extractor) 


## Acknowledgements

Some codes are brought from [VideoX-Fun](https://github.com/aigc-apps/VideoX-Fun/), and [LVCD](https://github.com/luckyhzt/LVCD). Thanks for their contributions~

If you have any questions, you can contact rainbowow@sjtu.edu.cn.


### Citation
If you find our work useful, please consider citing us:
```
@article{zhang2025animecolor,
      title={AnimeColor: Reference-based Animation Colorization with Diffusion Transformers}, 
      author={Yuhong Zhang and Liyao Wang and Han Wang and Danni Wu and Zuzeng Lin and Feng Wang and Li Song},
      journal={arXiv preprint arXiv:2507.20158},
      year={2025}
}

```
