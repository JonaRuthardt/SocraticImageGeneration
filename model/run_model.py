import argparse
from model.pipeline import Pipeline
from model.image_generator import ImageGeneratorType
from model.image_captioning import CaptioningModelType
from model.language_model import LanguageModelType
from utils.parrallel_arg_parsers import ParallelArgsParser
from data.data import DatasetType

import torch

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    precision = torch.float16 if torch.cuda.is_available() else torch.float32
    
    main_parser = ParallelArgsParser()
    # General
    p1 = main_parser.add_parser('pipeline')
    p1.add_argument('--experiment_name', default='default-experiment', type=str, help='Name of experiment')
    p1.add_argument('--max_cycles', default=5, type=int, help='Maximum number of times to optimize prompt and generate image')
    p1.add_argument('--terminate_on_similarity', default=False, type=bool, help="Whether to terminate the generation process when the language model regards the generated image and the original prompt as similar enough")
    
    # Dataset
    p2 = main_parser.add_parser('dataset')
    p2.add_argument('--dataset', default=None, type=str, choices=[d.value for d in DatasetType], help='Dataset to get prompts from')
    p2.add_argument('--prompt', default=None, type=str)
    #TODO add dataset-specific arguments

    # Image generator
    p3 = main_parser.add_parser('image_generator')
    p3.add_argument('--model', default=ImageGeneratorType.StableDiffusionV1_4.value, type=str, choices=[m.value for m in ImageGeneratorType], help='Image generator model')
    p3.add_argument('--device_map', default=device, type=str, help="Device to run image generator on.")
    p3.add_argument('--torch_dtype', default=precision, type=torch.dtype, help="Model Precision.")
    p3.add_argument('--seed', default=None, type=int, help="The seed for the model.")

    # Image captioning
    p4 = main_parser.add_parser('image_captioning')
    p4.add_argument('--device_map',
                    default='cuda', 
                    help='Image captioning model')
    p4.add_argument('--model', default=CaptioningModelType.BLIP_LARGE.value, type=str, choices=[m.value for m in CaptioningModelType], help='Image captioning model')

    # Language model
    p5 = main_parser.add_parser('language_model')
    p5.add_argument('--model', default=LanguageModelType.chat_gpt.value, type=str, choices=[m.value for m in LanguageModelType], help='Language model')
    #TODO add language model-specific arguments
                        
    args = main_parser.parse_args()
    args = {k:vars(v) for k,v in vars(args).items()}
    pipeline = Pipeline(**args)

if __name__ == '__main__':
    main()