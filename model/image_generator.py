import os, sys

def load_image_generator(model: str, config_file: str = None):
    """
    Load specified image generator model

    Parameters:
        model (str): name of model to load
        config_file (str): path to config file for model
    Returns:
        ImageGenerator: instanciated and configured image generator sub-class
    """

    # TODO instanciate image generator sub-class from given model name

    raise NotImplementedError

class ImageGenerator:
    """
    Base class for image generators
    """

    def __init__(self):
        raise NotImplementedError
    
    def generate_image(self, prompt: str):
        """
        Generate image given prompt

        Parameters:
            prompt (str): prompt to generate image from
        Returns:
            str: path to generated image #TODO: do we want to return the image itself?
        """

        #TODO generate image from prompt

        raise NotImplementedError