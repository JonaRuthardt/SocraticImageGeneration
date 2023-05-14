import os, sys
from PIL import Image
import enum
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration, AutoProcessor, Blip2ForConditionalGeneration

class CaptioningModelType(enum.Enum):
    BLIP_LARGE = "blip_large"
    BLIP2 = "blip_2"

def load_captioning_model(**kwargs):
    """
    Load specified image captioning model

    Parameters:
        kwargs (dict): additional config arguments to pass to model
    Returns:
        CaptioningModel: instanciated and configured captioning model sub-class
    """
    print("Loading captioning model")

    model_name = kwargs.get('model', CaptioningModelType.BLIP2.value)
    kwargs["model"] = model_name
    if model_name == CaptioningModelType.BLIP_LARGE.value:
        captioning_model = BlipLarge(**kwargs)
    elif model_name == CaptioningModelType.BLIP2.value:
        captioning_model = Blip2(**kwargs)
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
        
        out = self.model.generate(**inputs, max_new_tokens=150)
        caption = self.processor.decode(out[0], skip_special_tokens=True)

        return caption

class Blip2(CaptioningModel):
    """
    Blip-based image captioning model
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.device = kwargs["device_map"]
        self.processor = AutoProcessor.from_pretrained("Salesforce/blip2-opt-2.7b")
        self.model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b", torch_dtype=torch.float16).to(self.device)

    def generate_caption(self, image: Image):
        """
        Generate caption for image
        
        Parameters:
            image (PIL.Image): image to generate caption for
            cap_text (str) [optional]: caption text to condition on
        Returns:
            str: generated caption
        """
        inputs = self.processor(image, return_tensors="pt").to(self.device, torch.float16)
        generated_ids = self.model.generate(**inputs, max_new_tokens=20)
        caption = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        return caption

