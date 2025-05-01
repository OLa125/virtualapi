# inference_api.py

import os
import torch
from omegaconf import OmegaConf
from diffusers import AutoencoderKL, DDPMScheduler
from transformers import (
    CLIPTextModel, 
    CLIPTokenizer, 
    CLIPTextModelWithProjection
)
from promptdresser.models.unet import UNet2DConditionModel
from promptdresser.models.cloth_encoder import ClothEncoder
from promptdresser.pipelines.sdxl import PromptDresser

class PromptDresserPipeline:
    def __init__(self, config_path: str):
        self.args = self._load_args(config_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.weight_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        self._load_models()

    def _load_args(self, config_path):
        class Args:
            config_p = config_path
            interm_cloth_start_ratio = None
            timestep_spacing = "leading"
            no_zero_snr = False
            mixed_precision = "fp16"
            init_model_path = "./pretrained_models/stable-diffusion-xl-1.0-inpainting-0.1"
            init_vae_path = "./pretrained_models/sdxl-vae-fp16-fix"
            init_cloth_encoder_path = "./pretrained_models/stable-diffusion-xl-base-1.0"
        return Args()

    def _load_models(self):
        config = OmegaConf.load(self.args.config_p)
        if self.args.interm_cloth_start_ratio is not None:
            config.interm_cloth_start_ratio = self.args.interm_cloth_start_ratio

        self.noise_scheduler = DDPMScheduler.from_pretrained(
            self.args.init_model_path, subfolder="scheduler",
            rescale_betas_zero_snr=not self.args.no_zero_snr,
            timestep_spacing=self.args.timestep_spacing
        )
        self.tokenizer = CLIPTokenizer.from_pretrained(self.args.init_model_path, subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained(self.args.init_model_path, subfolder="text_encoder")
        self.tokenizer_2 = CLIPTokenizer.from_pretrained(self.args.init_model_path, subfolder="tokenizer_2")
        self.text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(self.args.init_model_path, subfolder="text_encoder_2")
        self.vae = AutoencoderKL.from_pretrained(self.args.init_vae_path)
        self.unet = UNet2DConditionModel.from_pretrained(self.args.init_model_path, subfolder="unet")
        self.cloth_encoder = ClothEncoder.from_pretrained(self.args.init_cloth_encoder_path, subfolder="unet")

        # Disable gradients and send to device
        for model in [self.unet, self.vae, self.text_encoder, self.text_encoder_2, self.cloth_encoder]:
            model.requires_grad_(False)
            model.to(self.device, dtype=self.weight_dtype)

        self.pipeline = PromptDresser(
            config=config,
            device=self.device,
            noise_scheduler=self.noise_scheduler,
            tokenizer=self.tokenizer,
            text_encoder=self.text_encoder,
            tokenizer_2=self.tokenizer_2,
            text_encoder_2=self.text_encoder_2,
            vae=self.vae,
            unet=self.unet,
            cloth_encoder=self.cloth_encoder
        )

    def generate(self, inputs: dict):
        """
        Run the pipeline generation.
        inputs: dict should contain all the required keys depending on your PromptDresser pipeline
        """
        return self.pipeline(inputs)
