import os, sys
from PIL import Image
import enum

class CaptioningModelType(enum.Enum):
    ClipCap = "ClipCap"
    #TODO specify all available captioning models here

def load_captioning_model(model: str, **kwargs):
    """
    Load specified image captioning model

    Parameters:
        model (str): name of model to load
        kwargs (dict): additional config arguments to pass to model
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
    
    def generate_caption(self, image: Image):
        """
        Generate caption for image

        Parameters:
            image (PIL.Image): image to generate caption for
        Returns:
            str: generated caption
        """

        #TODO generate caption for image

        raise NotImplementedError