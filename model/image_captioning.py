import os, sys
from PIL import Image
import enum
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration

class CaptioningModelType(enum.Enum):
    BLIP_LARGE = "blip_large"
    #TODO specify all available captioning models here

def load_captioning_model(**kwargs):
    """
    Load specified image captioning model

    Parameters:
        kwargs (dict): additional config arguments to pass to model
    Returns:
        CaptioningModel: instanciated and configured captioning model sub-class
    """
    print("Loading captioning model")

    model_name = kwargs.get('model', CaptioningModelType.BLIP_LARGE.value)
    kwargs["model"] = model_name
    if model_name == CaptioningModelType.BLIP_LARGE.value:
        captioning_model = BlipLarge(**kwargs)
    else:
        raise ValueError(f"Unknown captioning model {model_name}")

    return captioning_model

class CaptioningModel:
    """
    Base class for image captioning models
    """

    def __init__(self, **kwargs):
        self.__model_name = kwargs["model"]

    @property
    def model_name(self):
        return self.__model_name
    
    @property
    def device(self) -> torch.device:
        """
        Get current model's device.
        """
        return self.model.device

    def generate_caption(self, image: Image, cap_text: str = '') -> str:
        """
        Generate caption for image

        Parameters:
            image (PIL.Image): image to generate caption for
            cap_text (str) [optional]: caption text to condition on
        Returns:
            str: generated caption
        """

        raise NotImplementedError
    
    def reset(self) -> None:
        """
        Optional method to reset model between generations
        """
        pass


class BlipLarge(CaptioningModel):
    """
    Blip-based image captioning model
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
        self.model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large").to(kwargs["device_map"])

    def generate_caption(self, image: Image, cap_text: str = ''):
        """
        Generate caption for image
        
        Parameters:
            image (PIL.Image): image to generate caption for
            cap_text (str) [optional]: caption text to condition on
        Returns:
            str: generated caption
        """
        if cap_text:
            inputs = self.processor(image, cap_text, return_tensors="pt").to(self.device)
        else:
            inputs = self.processor(image, return_tensors="pt").to(self.device)
        
        out = self.model.generate(**inputs, max_new_tokens=100)
        caption = self.processor.decode(out[0], skip_special_tokens=True)

        return caption

