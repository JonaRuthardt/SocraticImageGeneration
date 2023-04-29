import os, sys
import enum

class ImageGeneratorType(enum.Enum):
    StableDiffusion = "StableDiffusion"
    #TODO specify all available image generators here

def load_image_generator(model: str, **kwargs):
    """
    Load specified image generator model

    Parameters:
        model (str): name of model to load
        kwargs (dict): additional config arguments to pass to model
    Returns:
        ImageGenerator: instanciated and configured image generator sub-class
    """

    # TODO instanciate image generator sub-class from given model name

    raise NotImplementedError

class ImageGenerator:
    """
    Base class for image generators
    """

    def __init__(self, **kwargs):
        raise NotImplementedError
    
    def generate_image(self, prompt: str):
        """
        Generate image given prompt

        Parameters:
            prompt (str): prompt to generate image from
        Returns:
            PIL.Image: generated image
        """

        #TODO generate image from prompt

        raise NotImplementedError
    
    def reset(self):
        """
        Optional method to reset model between generations
        """
        pass