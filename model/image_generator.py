import os, sys

# Libraries for types
from typing import Optional, Union, Dict
import enum
from abc import ABC, abstractmethod
import PIL

# DL Libraries
import torch
from diffusers import StableDiffusionPipeline
from PIL import Image

class ImageGeneratorType(enum.Enum):
    StableDiffusionV1_4 = "CompVis/stable-diffusion-v1-4"
    StableDiffusionV1_5 = "runwayml/stable-diffusion-v1-5"
    StableDiffusionV2_1_Base = "stabilityai/stable-diffusion-2-1-base"
    StableDiffusionV2_1 ="stabilityai/stable-diffusion-2-1"

IMAGE_GENERATORS = frozenset(set([img_gen_t.value for img_gen_t in ImageGeneratorType]))

def load_image_generator(image_generator: str,  #TODO fix
                         device_map: Optional[Union[str, torch.device]]=None, 
                         torch_dtype:str|torch.dtype=torch.float16,
                         seed:int=None, 
                         **kwargs):
    """
    Load specified image generator model

    Parameters:
        model (str): name of model to load
        kwargs (dict): additional config arguments to pass to model
    Returns:
        ImageGenerator: instanciated and configured image generator sub-class
    """
    print("Loading image generation model")

    model = image_generator
    # Only the allowed types can be used
    if model not in IMAGE_GENERATORS:
        raise f"The Text-to-Image Generator given (\"{model}\") is not a recognised Generator from: {IMAGE_GENERATORS}."
    return StableDiffuser(model, device_map, torch_dtype=torch_dtype, seed=seed, **kwargs)

class ImageGenerator(ABC):
    """
    Base Abstract class for image generators
    """
    
    def __init__(self, model: str, **kwargs):
        self.__model_name = model

    @abstractmethod
    def generate_image(self, prompt: str):
        """
        Generate image given prompt

        Parameters:
            prompt (str): prompt to generate image from
        Returns:
            PIL.Image: generated image
        """
        pass
    
    @property
    def model_name(self):
        return self.__model_name

    @property
    def device(self) -> torch.device:
        """
        Get current model's device.
        """
        return torch.device('cpu')

    @abstractmethod
    def to(self, torch_device: Optional[Union[str, torch.device]]=None) -> torch.device:
        """
        Chnge the device of the current model
        """
        pass

    @abstractmethod
    def reset(self, seed:int=None):
        """
        Optional method to reset model between generations
        """
        if seed is not None:
            self.__seed = seed
        if self.__seed is not None:
             self.__reset_generator()

class StableDiffuser(ImageGenerator):
    """
    Diffusers class as image generators
    """
    def __init__(self, model: str,  
                 device_map: Optional[Union[str, torch.device]]=None, 
                 torch_dtype:str|torch.dtype=torch.float16, 
                 seed:int=None, 
                 **kwargs):
        ImageGenerator.__init__(self,model)
        ### Other Parameters of the Diffusers to keep in mind ###
        # 1) num_inference_steps : (default=50) --> results are better the more steps you use but becomes slower
        # 2) guidance_scale : () --> It is a way to increase the adherence to the conditional signal which in 
        #   this case is text as well as overall sample quality. In simple terms classifier free guidance 
        #   forces the generation to better match with the prompt. Numbers like 7 or 8.5 give good results, 
        #   if you use a very large number the images might look good, but will be less diverse.
        self.__model = StableDiffusionPipeline.from_pretrained(model, safety_checker = None,
                                                                torch_dtype=torch_dtype)
        self.__model = self.__model.to(device_map)
        self.__seed = seed
        if self.__seed is not None:
            self.__reset_generator()
        else:
            self.__generator = None
    @property
    def model(self):
        return self.__model

    @property
    def device(self) -> torch.device:
        """
        Get current model's device.
        """
        return self.__model.device

    def __reset_generator(self):
        self.__generator = torch.Generator(self.__model.device).manual_seed(self.__seed) 

    def generate_image(self, prompt: str, num_inference_steps:int=50, guidance_scale:float=7.5,
                       height:int=512, width:int=512, **kwargs) -> Image:
        """
        Generate image given prompt

        Parameters:
            prompt (str): prompt to generate image from
            height (int): the height of  the image to generate. Must be multiple of 8.
            width (int): the width of  the image to generate. Must be multiple of 8.
        Returns:
            PIL.Image: generated image
        """
        image = self.__model(prompt, 
                        generator=self.__generator,
                        num_inference_steps=num_inference_steps,
                        guidance_scale=guidance_scale, 
                        height=height, 
                        width=width).images[0]
        return image

    def to(self, torch_device: Optional[Union[str, torch.device]]=None) -> torch.device:
        """
        Change the device of the current model
        """
        self.__model = self.__model.to(torch_device)

    def reset(self, seed:int=None):
        """
        Optional method to reset model between generations
        """
        if seed is not None:
            self.__seed = seed
        if self.__seed is not None:
             self.__reset_generator()
