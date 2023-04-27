import pytest
from PIL import Image
from model.image_captioning import CaptioningModel, CaptioningModelType

@pytest.fixture
def image():
    return Image.open("data/test/cat.jpeg")

@pytest.fixture
def captioning():
    return CaptioningModel(CaptioningModelType.BLIP_LARGE.value)

def test_image_captioning(captioning):
    assert len(captioning.generate_caption(image, 'a photograph of'))
    assert len(captioning.generate_caption(image))
