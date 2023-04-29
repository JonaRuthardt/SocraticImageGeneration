import os, sys
import random
import json

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

        self.language_model = load_language_model(**kwargs)
        self.image_generator = load_image_generator(**kwargs)
        self.image_captioning = load_captioning_model(**kwargs)

        self.terminate_on_similarity = kwargs.get("terminate_on_similarity", True)

        # Set-up folder to store generated images and save hyperparameters
        self.image_id = 0
        experiment_name = kwargs.get("experiment_name", "default-experiment")
        self.path = os.path.join("data/results", experiment_name)
        os.mkdir(self.path, exist_ok=False)
        with open(os.path.join(self.path, "hyperparameters.json"), "w") as f:
            json.dump(self.hyperparameters, f)

        self.dataset = kwargs.get("dataset", None)
        if self.dataset is not None:
            # Load and configure dataset
            if self.dataset == "parti-prompts":
                self.dataset = datasets.load_dataset("parti_prompts", split="train")["Prompt"]
            elif self.dataset == "flickr30k":
                dataset = datasets.load_dataset("embedding-data/flickr30k-captions", split="train")
                self.dataset = [d[0] for d in dataset["set"]]
            else:
                raise ValueError(f"Unknown dataset {self.dataset}")
            
            # Execute dataset-based experiment pipeline
            #TODO do we always directly want to execute it here or implement running the experiments outside of the pipeline?
            self.generate_images_from_dataset()

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
        os.mkdir(folder_name, exist_ok=False)
        self.image_id += 1

        # Generate image
        prompt = user_prompt
        previous_prompts = []
        for i in range(max_cycles):
            # Generate image
            image = self.image_generator.generate_image(prompt)

            # Generate caption
            caption = self.image_captioning.generate_caption(image)

            # Check termnation condition
            if self.terminate_on_similarity and self.language_model.check_similarity(prompt, caption):
                break

            # Optimize prompt
            prompt = self.language_model.generate_optimized_prompt(user_prompt, caption, previous_prompts)
            previous_prompts.append(prompt)

            # Save image and caption
            image.save(os.path.join(folder_name, f"image_{i}.png"))
            with open(os.path.join(folder_name, f"caption_{i}.txt"), "w") as f:
                f.write(caption)
            
        # Store intermediate and final prompts
        with open(os.path.join(folder_name, "prompts.csv"), "w") as f:
            f.write(f"user_prompt\t{user_prompt}\n")
            for i, prompt in enumerate(previous_prompts):
                f.write(f"optimized_prompt_{i}\t{prompt}\n")
        
        # Return path to folder of generated images
        return folder_name
    
    def generate_images_from_dataset(self, max_cycles: int = 5):
        """
        Generate images based on prompts from dataset
        """

        for prompt in self.dataset:
            self.generate_image(prompt, max_cycles=max_cycles)
            self.reset_pipeline()
    
    def reset_pipeline(self):
        """
        Reset pipeline and models between images
        """
        self.language_model.reset()
        self.image_generator.reset()
        self.image_captioning.reset()
    
