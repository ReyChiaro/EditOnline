# 🪄 EditOnline
A WebUI for image editing, support mask-based and instruction-based editing.


## Quick Start

> Python 3.14, diffusers 0.37.1 and torch 2.10 are preferred.

**Environment Configurations**: This project use [uv](https://docs.astral.sh/uv/getting-started/installation/) to build the environment. The configuration file have been provided, see `pyproject.toml` for more details. Run
```sh
uv sync
# Or uv sync --index-url other-source to enable synchronizing the environment from the specific source.
```
and the environment will be created in `.venv` by default. To modify the path of the environment, specify the env variable `UV_PROJECT_ENVIRONMENT` to your prefer path.


**DiT and Pre-trained weights**: The base model is [QwenImage-Edit-2511](https://huggingface.co/Qwen/Qwen-Image-Edit-2511/tree/main), we provide a pre-trained LoRA for mask-based image editing. The weights will be released.

- To modify the path which the model will be loaded from, you can look [config](editonline/config.py) for more details.


**External repo**: The mask generator depends to external repository, [SAM](https://github.com/facebookresearch/segment-anything.git) is used by default to generate more accurate mask. You add another segmentation models in `mask_generator/model_hub`.





## 🤓☝️ TODO

- [x] Release EditOnline framework
- [ ] Release mask-based editing LoRA weights
- [ ] Support API query