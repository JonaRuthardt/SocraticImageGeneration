import argparse
from model.pipeline import Pipeline
from model.image_generator import ImageGeneratorType
from model.image_captioning import CaptioningModelType
from model.language_model import LanguageModelType

from data.data import DatasetType

def main():
    parser = argparse.ArgumentParser()

    # General
    parser.add_argument('--experiment_name', default='default-experiment', type=str, help='Name of experiment')
    parser.add_argument('--max_cycles', default=3, type=int, help='Maximum number of times to optimize prompt and generate image')
    parser.add_argument('--terminate_on_similarity', default=False, type=bool, help="Whether to terminate the generation process when the language model regards the generated image and the original prompt as similar enough")
    parser.add_argument('--device_map', default="cuda", type=str, help="Device to run models on")
    
    # Dataset
    parser.add_argument('--dataset', default=None, type=str, choices=[d.value for d in DatasetType], help='Dataset to get prompts from')
    parser.add_argument('--prompt', default=None, type=str)
    #TODO add dataset-specific arguments

    # Image generator
    parser.add_argument('--image_generator', default=ImageGeneratorType.StableDiffusionV2_1.value, type=str, choices=[m.value for m in ImageGeneratorType], help='Image generator model')
    #TODO add image generator-specific arguments

    # Image captioning
    parser.add_argument('--image_captioning', default=CaptioningModelType.BLIP_LARGE.value, type=str, choices=[m.value for m in CaptioningModelType], help='Image captioning model')
    #TODO add image captioning-specific arguments

    # Language model
    parser.add_argument('--language_model', default=LanguageModelType.chat_gpt.value, type=str, choices=[m.value for m in LanguageModelType], help='Language model')
    #TODO add language model-specific arguments
                        
    args = parser.parse_args()

    pipeline = Pipeline(**vars(args))

if __name__ == '__main__':
    main()