import os, sys

def load_captioning_model(model: str, config_file: str = None):
    """
    Load specified image captioning model

    Parameters:
        model (str): name of model to load
        config_file (str): path to config file for model
    Returns:
        CaptioningModel: instanciated and configured captioning model sub-class
    """

    # TODO instanciate captioning model sub-class from given model name

    raise NotImplementedError

class CaptioningModel:
    """
    Base class for image captioning models
    """

    def __init__(self):
        raise NotImplementedError
    
    def generate_caption(self, image_path: str):
        """
        Generate caption for image

        Parameters:
            image_path (str): path to image to generate caption for #TODO do we want to pass the image itself?
        Returns:
            str: generated caption
        """

        #TODO generate caption for image

        raise NotImplementedError