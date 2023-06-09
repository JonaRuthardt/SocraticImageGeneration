import os, sys
import random
import json
import time
import pandas as pd

import datasets

from model.language_model import load_language_model
from model.image_generator import load_image_generator
from model.image_captioning import load_captioning_model
from PIL import Image

random.seed(42)

class Pipeline:
    def __init__(self, **kwargs):
        """
        Initialize pipeline
        """
        self.hyperparameters = kwargs

        self.language_model = load_language_model(**kwargs.get('language_model',{}))
        self.image_generator = load_image_generator(**kwargs.get('image_generator',{}))
        self.image_captioning = load_captioning_model(**kwargs.get('image_captioning',{}))

        self.pipeline_mode = kwargs.get('pipeline',{}).get("mode", "inference")
        self.terminate_on_similarity = kwargs.get('pipeline',{}).get("terminate_on_similarity", True)
        self.select_best_image = kwargs.get('pipeline',{}).get("select_best_image", True)
        self.demo = kwargs.get('pipeline',{}).get("demo", False)
        # Set-up folder to store generated images and save hyperparameters
        self.image_id = 0
        experiment_name = kwargs.get('pipeline',{}).get("experiment_name", "default-experiment")
        self.path = os.path.join("data/results", experiment_name)
        os.makedirs(self.path, exist_ok=True)
        with open(os.path.join(self.path, "hyperparameters.json"), "w") as f:
            def convert_dict2str(dict):
                new_dict = {}
                for key, values in dict.items():
                    if type(values) == type(dict):
                        new_dict[key] = convert_dict2str(values)
                    else:
                        new_dict[key] = str(values)
                return new_dict
                
            json.dump(convert_dict2str(self.hyperparameters), f, sort_keys=True, indent=4)

        self.dataset_name = kwargs.get('dataset',{}).get("dataset", None)
        self.dataset = None
        if self.dataset_name is not None:
            # Load and configure dataset
            if self.dataset_name == "parti-prompts":
                self.dataset = datasets.load_dataset("nateraw/parti-prompts", split="train")["Prompt"]
            elif self.dataset_name == "parti-prompts-small":
                self.dataset = datasets.load_dataset("nateraw/parti-prompts", split="train")["Prompt"]
                self.dataset = [self.dataset[i] for i in range(0, len(self.dataset), len(self.dataset)//50)]
            elif self.dataset_name == "parti-prompts-medium":
                self.dataset = datasets.load_dataset("nateraw/parti-prompts", split="train")["Prompt"]
                self.dataset = [self.dataset[i] for i in range(0, len(self.dataset), len(self.dataset)//200)]
            elif self.dataset_name == "flickr30k":
                dataset = datasets.load_dataset("embedding-data/flickr30k-captions", split="train")
                self.dataset = [d[0] for d in dataset["set"]]
            elif self.dataset_name == "flickr30k-small":
                dataset = datasets.load_dataset("embedding-data/flickr30k-captions", split="train")
                self.dataset = [d[0] for d in dataset["set"]]
                self.dataset = [self.dataset[i] for i in random.sample(range(len(self.dataset)), 50)]
            elif self.dataset_name == "cococaption-small":
                annotations = pd.read_csv("data/datasets/coco-small/annotations.tsv", sep="\t")
                self.dataset = annotations["caption 1"].tolist()
                self.original_images = ("data/datasets/coco-small/" + annotations["file_name"]).tolist()
            elif self.dataset_name == "cococaption-medium":
                annotations = pd.read_csv("data/datasets/coco-medium/annotations.tsv", sep="\t")
                self.dataset = annotations["caption 1"].tolist()
                self.original_images = ("data/datasets/coco-medium/" + annotations["file_name"]).tolist()
            elif self.dataset_name == "cococaption-large":
                annotations = pd.read_csv("data/datasets/coco-large/annotations.tsv", sep="\t")
                self.dataset = annotations["caption 1"].tolist()
                self.original_images = ("data/datasets/coco-large/" + annotations["file_name"]).tolist()
            else:
                raise ValueError(f"Unknown dataset {self.dataset_name}")

            # Execute dataset-based experiment pipeline
            self.generate_images_from_dataset(max_cycles=kwargs["pipeline"]["max_cycles"])
        elif kwargs.get('dataset',{}).get("prompt", None) is not None:
            # Execute single prompt experiment pipeline
            self.generate_image(kwargs["dataset"]["prompt"], max_cycles=kwargs["pipeline"]["max_cycles"])

    def generate_image(self, user_prompt: str, max_cycles: int = 5):
        """
        Generate image given prompt

        Parameters:
            user_prompt (str): (user) prompt to generate image from
            max_cycles (int): maximum number of times to optimize prompt and generate image
        Returns:
            str: path to folder of generated image(s)
        """
        
        # Set up folder to store generated images
        folder_name = str(self.image_id).zfill(6)
        folder_name = os.path.join(self.path, folder_name)
        self.image_id += 1
        os.makedirs(folder_name, exist_ok=True)

        # Generate image
        original_prompt = user_prompt
        prompt = user_prompt
        captions = []
        previous_prompts = []
        terminated = -1
        best_image_idx = -1

        # Generate and save image for original user prompt
        image = self.image_generator.generate_image(prompt)
        image.save(os.path.join(folder_name, f"image_{0}.png"))
        if self.demo:
            image_to_show = Image.open(os.path.join(folder_name, f"image_{0}.png"))
            image_to_show.show()

        for i in range(max_cycles):

            # Generate caption
            caption = self.image_captioning.generate_caption(image)
            captions.append(caption)

            # Check termination condition
            if self.terminate_on_similarity and self.language_model.check_similarity(original_prompt, caption) and terminated == -1:
                terminated = i
                if self.pipeline_mode != "full_experiment":
                    break

            # Optimize prompt
            prompt = self.language_model.generate_optimized_prompt(user_prompt, caption, previous_prompts)
            previous_prompts.append(prompt)

            # Generate and save image for optimized prompt
            image = self.image_generator.generate_image(prompt)
            image.save(os.path.join(folder_name, f"image_{i+1}.png"))
            if self.demo:
                image_to_show = Image.open(os.path.join(folder_name, f"image_{i+1}.png"))
                image_to_show.show()

        # Generate caption for final image
        if terminated == -1 or self.pipeline_mode == "full_experiment":
            caption = self.image_captioning.generate_caption(image)
            captions.append(caption)

        if self.select_best_image:
            best_image_idx = self.language_model.select_best_image(user_prompt, captions)

        # Store intermediate and final prompts and captions
        with open(os.path.join(folder_name, "prompts.csv"), "wb") as f:
            f.write(f"user_prompt\t{user_prompt}\n".encode("utf-8", errors="replace"))
            for i, prompt in enumerate(previous_prompts):
                f.write(f"optimized_prompt_{i}\t{prompt}\n".encode("utf-8", errors="replace"))
            if self.demo:
                print(f"optimized_prompt_{i}\t{prompt}\n")
        with open(os.path.join(folder_name, f"captions.csv"), "wb") as f:
            for i, caption in enumerate(captions):
                f.write(f"{i}\t{caption}\n".encode("utf-8", errors="replace"))
            if self.demo:
                print(f"{i}\t{caption}\n")
        with open(os.path.join(folder_name, f"results.txt"), "wb") as f:
            f.write(f"terminated at iteration:\t{terminated}\n".encode("utf-8", errors="replace"))
            f.write(f"best image:\t{best_image_idx}\n".encode("utf-8", errors="replace"))
        
        # Return path to folder of generated images
        return folder_name
    
    def generate_images_from_dataset(self, max_cycles: int = 5):
        """
        Generate images based on prompts from dataset
        """

        for i, prompt in enumerate(self.dataset):
            # Save orignal image
            folder_name = str(i).zfill(6)
            folder_name = os.path.join(self.path, folder_name)
            os.makedirs(folder_name, exist_ok=False)
            if self.dataset_name in ["cococaption-small", "cococaption-medium", "cococaption-large"]:
                original_image = Image.open(os.path.join(self.original_images[i]))
                original_image.save(os.path.join(folder_name, f"original_image.png"))
            start = time.time()
            try:
                self.generate_image(prompt, max_cycles=max_cycles)
            except OSError as e:
                # image was already generated
                print(e)
                #TODO remove or implement nicer
                pass
            self.reset_pipeline()
            print(f"Time for prompt {i}: {round(time.time() - start, 2)}s")
    
    def reset_pipeline(self):
        """
        Reset pipeline and models between images
        """
        self.language_model.reset()
        self.image_generator.reset()
        self.image_captioning.reset()
    
