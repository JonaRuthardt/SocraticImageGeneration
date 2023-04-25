import os, sys
import random
import json

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
        experiment_name = kwargs.get("experiment_name", "default-experiment")
        self.path = os.path.join("data", experiment_name)
        os.mkdir(self.path, exist_ok=False)
        with open(os.path.join(self.path, "hyperparameters.json"), "w") as f:
            json.dump(kwargs, f)


        # TODO potentially load other models or configurations
        raise NotImplementedError

    def generate_image(self, user_prompt: str, max_cycles: int = 5):
        """
        Generate image given prompt

        Parameters:
            user_prompt (str): (user) prompt to generate image from
            max_cycles (int): maximum number of times to optimize prompt and generate image
        Returns:
            str: path to folder of generated image(s)
        """
        
        # Setup folder to store generated images
        #TODO could be implemented differently in a better way
        #BUG what if you re-run with same prompts? Won't overwrite the data but will create a new folder
        folder_name = str(random.randint(0, 999999)).zfill(6)
        while os.path.exists(os.path.join(self.path, folder_name)):
            folder_name = str(random.randint(0, 999999)).zfill(6)
        folder_name = os.path.join("data", folder_name)
        os.mkdir(folder_name)

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
        with open(os.path.join(folder_name, "prompts.txt"), "w") as f:
            f.write("<USER PROMPT>\n" + user_prompt + "\n\n")
            for i, prompt in enumerate(previous_prompts):
                f.write(f"<OPTIMIZED PROMPT ITERATION {i}>\n" + prompt + "\n\n")
        
        # Return path to folder of generated images
        return folder_name
    
    def reset_pipeline(self):
        """
        Reset pipeline and models between images
        """
        self.language_model.reset()
        self.image_generator.reset()
        self.image_captioning.reset()
    
