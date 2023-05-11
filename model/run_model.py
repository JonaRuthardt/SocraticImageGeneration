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
    p1.add_argument('--mode', default='inference', type=str, choices=['inference', 'full_experiment'], help='Mode of pipeline')
    p1.add_argument('--experiment_name', default='default-experiment', type=str, help='Name of experiment')
    p1.add_argument('--max_cycles', default=5, type=int, help='Maximum number of times to optimize prompt and generate image')
    p1.add_argument('--terminate_on_similarity', default=False, type=bool, help="Whether to terminate the generation process when the language model regards the generated image and the original prompt as similar enough")
    p1.add_argument('--select_best_image', default=False, type=bool, help="Whether to select the best image from the generated images")
    
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
    p3.add_argument('--num_inference_steps', default=50, type=int, help="Number of inference steps. The more the better results but more execution time.")
    p3.add_argument('--guidance_scale', default=7.5, type=float, help="Classifier free guidance forces the generation to better match with the prompt.")
    p3.add_argument('--height', default=512, type=int, help="Height of the image to be generated.")
    p3.add_argument('--width', default=512, type=int, help="Width of the image to be generated.")

    # Image captioning
    p4 = main_parser.add_parser('image_captioning')
    p4.add_argument('--device_map',
                    default=device,
                    help='Image captioning model')
    p4.add_argument('--model', default=CaptioningModelType.BLIP_LARGE.value, type=str, choices=[m.value for m in CaptioningModelType], help='Image captioning model')

    # Language model
    p5 = main_parser.add_parser('language_model')
    p5.add_argument('--model', default=LanguageModelType.chat_gpt.value, type=str, choices=[m.value for m in LanguageModelType], help='Language model')
    p5.add_argument('--api_key_path', default="config/openai_api_key.txt", type=str, help='Path to text file containing api key')
    p5.add_argument('--template', default='config/templates/default_template.txt', type=str, help='Path to template to use for language model')
    p5.add_argument('--system_prompt', default='config/templates/model_role.txt', type=str, help='Path to system prompt to use for language model')
    p5.add_argument('--similarity_template', default='config/templates/default_similarity_template.txt', type=str, help='Path to template to use for similarity check')
    p5.add_argument('--system_sim_prompt', default='config/templates/model_role_similarity.txt', type=str, help='Path to system prompt to use for similarity check')
    p5.add_argument('--best_image_template', default='config/templates/default_best_image_template.txt', type=str, help='Path to template to use for best image selection')
    p5.add_argument('--system_best_image_prompt', default='config/templates/model_role_best_image.txt', type=str, help='Path to system prompt to use for best image selection')
                        
    args = main_parser.parse_args()
    args = {k:vars(v) for k,v in vars(args).items()}
    pipeline = Pipeline(**args)

if __name__ == '__main__':
    main()