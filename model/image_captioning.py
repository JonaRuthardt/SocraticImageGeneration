import os, sys
from PIL import Image
import enum
from transformers import BlipProcessor, BlipForConditionalGeneration

class CaptioningModelType(enum.Enum):
    BLIP_LARGE = "Salesforce/blip-image-captioning-large"
    #TODO specify all available captioning models here

def load_captioning_model(**kwargs):
    """
    Load specified image captioning model

    Parameters:
        kwargs (dict): additional config arguments to pass to model
    Returns:
        CaptioningModel: instanciated and configured captioning model sub-class
    """

    return CaptioningModel(kwargs['model_name'])

class CaptioningModel:
    """
    Base class for image captioning models
    """

    def __init__(self, model_name: str):
        self.processor = BlipProcessor.from_pretrained(model_name)
        self.model = BlipForConditionalGeneration.from_pretrained(model_name)
    
    def generate_caption(self, image: Image, cap_text: str = ''):
        """
        Generate caption for image

        Parameters:
            image (PIL.Image): image to generate caption for
        Returns:
            str: generated caption
        """

        if cap_text:
            inputs = self.processor(image, cap_text, return_tensors="pt")
        else:
            inputs = self.processor(image, return_tensors="pt")
        
        out = self.model.generate(**inputs)
        caption = self.processor.decode(out[0], skip_special_tokens=True)

        return caption
    
    def reset(self):
        """
        Optional method to reset model between generations
        """
        pass