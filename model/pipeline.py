import os, sys
import random
import json
import time

import datasets

from model.language_model import load_language_model
from model.image_generator import load_image_generator
from model.image_captioning import load_captioning_model


class Pipeline:
    def __init__(self, **kwargs):
        """
        Initialize pipeline
        """
        self.hyperparameters = kwargs

        self.language_model = load_language_model(**kwargs.get('language_model',{}))
        self.image_generator = load_image_generator(**kwargs.get('image_generator',{}))
        self.image_captioning = load_captioning_model(**kwargs.get('image_captioning',{}))

        self.terminate_on_similarity = kwargs.get('pipeline',{}).get("terminate_on_similarity", True)

        # Set-up folder to store generated images and save hyperparameters
        self.image_id = 0
        experiment_name = kwargs.get('pipeline',{}).get("experiment_name", "default-experiment")
        self.path = os.path.join("data/results", experiment_name)
        os.makedirs(self.path, exist_ok=False)
        with open(os.path.join(self.path, "hyperparameters.json"), "w") as f:
            json.dump(self.hyperparameters, f)

        self.dataset = kwargs.get('dataset',{}).get("dataset", None)
        if self.dataset is not None:
            # Load and configure dataset
            if self.dataset == "parti-prompts":
                self.dataset = datasets.load_dataset("nateraw/parti-prompts", split="train")["Prompt"]
            elif self.dataset == "flickr30k":
                dataset = datasets.load_dataset("embedding-data/flickr30k-captions", split="train")
                self.dataset = [d[0] for d in dataset["set"]]
            else:
                raise ValueError(f"Unknown dataset {self.dataset}")
            
            # Execute dataset-based experiment pipeline
            #TODO do we always directly want to execute it here or implement running the experiments outside of the pipeline?
            self.generate_images_from_dataset(max_cycles=kwargs["max_cycles"])

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
        os.makedirs(folder_name, exist_ok=False)
        self.image_id += 1

        # Generate image
        prompt = user_prompt
        captions = []
        previous_prompts = []

        # Generate and save image for original user prompt
        image = self.image_generator.generate_image(prompt)
        image.save(os.path.join(folder_name, f"image_{0}.png"))

        for i in range(max_cycles):

            # Generate caption
            caption = self.image_captioning.generate_caption(image)
            captions.append(caption)

            # Check termnation condition
            if self.terminate_on_similarity and self.language_model.check_similarity(prompt, caption):
                break

            # Optimize prompt
            prompt = self.language_model.generate_optimized_prompt(user_prompt, caption, previous_prompts)
            previous_prompts.append(prompt)

            # Generate and save image for optimized prompt
            image = self.image_generator.generate_image(prompt)
            image.save(os.path.join(folder_name, f"image_{i+1}.png"))

        # Store intermediate and final prompts and captions
        with open(os.path.join(folder_name, "prompts.csv"), "w") as f:
            f.write(f"user_prompt\t{user_prompt}\n")
            for i, prompt in enumerate(previous_prompts):
                f.write(f"optimized_prompt_{i}\t"+prompt+"\n")
        with open(os.path.join(folder_name, f"captions.csv"), "w") as f:
            for i, caption in enumerate(captions):
                f.write(f"{i}\t{caption}\n")
        
        # Return path to folder of generated images
        return folder_name
    
    def generate_images_from_dataset(self, max_cycles: int = 5):
        """
        Generate images based on prompts from dataset
        """

        for i, prompt in enumerate(self.dataset):
            start = time.time()
            self.generate_image(prompt, max_cycles=max_cycles)
            self.reset_pipeline()
            print(f"Time for prompt {i}: {round(time.time() - start, 2)}s")
    
    def reset_pipeline(self):
        """
        Reset pipeline and models between images
        """
        self.language_model.reset()
        self.image_generator.reset()
        self.image_captioning.reset()
    
