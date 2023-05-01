import os, sys
import json
import argparse
from tqdm import tqdm

import pandas as pd
import torch
from PIL import Image
import open_clip

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class Evaluate():
    """
    Base class for evaluation
    """
    def __init__(self, **kwargs) -> None:
        self.experiment_folder = os.path.join("data/results", kwargs.get("experiment_name", "default-experiment"))
        self.hyperparameters = self.load_hyperparameters()


    def load_hyperparameters(self):
        """
        Load hyperparameters from experiment folder
        """
        with open(os.path.join(self.experiment_folder, "hyperparameters.json"), "r") as f:
            return json.load(f)
        
    def load_prompts(self, folder: str):
        """
        Load prompts from given prompt-specific folder containing generated images
        """
        with open(os.path.join(folder, "prompts.csv"), "r") as f:
            orig_prompts = f.read().splitlines()
            prompts = []
            for prompt_line in orig_prompts:
                if prompt_line.startswith("optimized_prompt") or prompt_line.startswith("user_prompt"):
                    prompts.append(prompt_line.split("\t")[-1])
                else:
                    prompts[-1] += prompt_line
        

        return prompts
    
    def evaluate(self):
        """
        Evaluate generated image(s)
        """
        raise NotImplementedError
    
    def save_results(self):
        """
        Save evaluation results to file
        """
        pass
        
class CLIPScore(Evaluate):
    """
    Evaluate generated images using CLIPScore-based metrics
    """
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        
        # Load CLIP model
        self.model, _, self.preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
        self.model = self.model.to(device)
        self.tokenizer = open_clip.get_tokenizer('ViT-B-32')

        self.results_df = pd.DataFrame(columns=["prompt_id", "image_id", "score", "user_prompt", "optimized_prompt", "image_path"], index=["prompt_id", "image_id"])
        
    def evaluate(self):
        """
        Evaluate image using CLIP
        """
        # iterate over all prompts/generations of experiment
        for prompt_folder in tqdm(os.listdir(self.experiment_folder)):
            prompt_folder = os.path.join(self.experiment_folder, prompt_folder)
            if not os.path.isdir(prompt_folder):
                continue
            prompts = self.load_prompts(prompt_folder)

            # Tokenize and encode user prompt
            user_prompt = self.tokenizer([prompts[0]]).to(device)
            user_prompt_features = self.model.encode_text(user_prompt)
            user_prompt_features /= user_prompt_features.norm(dim=-1, keepdim=True)


            # iterate over all generated images
            images = []
            images_features = []
            for image_idx, prompt in enumerate(prompts):
                # load, preprocess, and encode image
                image = Image.open(os.path.join(prompt_folder, f"image_{image_idx}.png"))
                images.append(image)
                image = self.preprocess(image).unsqueeze(0).to(device)
                image_features = self.model.encode_image(image)
                image_features /= image_features.norm(dim=-1, keepdim=True)
                images_features.append(image_features)

                # calculate CLIP-based similarity of original prompt and generated image
                #TODO check if this is best and correct way to implement scoring (consider vectorization)
                score = (100.0 * image_features @ user_prompt_features.T).data.cpu().numpy().item()

                df_row = {
                    "prompt_id": int(prompt_folder.split("/")[-1]),
                    "image_id": image_idx,
                    "score": score,
                    "user_prompt": prompts[0],
                    "optimized_prompt": prompt,
                    "image_path": os.path.join(prompt_folder, f"image_{image_idx}.png"),
                }
                self.results_df = pd.concat([self.results_df, pd.DataFrame([df_row.values()], columns=self.results_df.columns)])

    def save_results(self):
        """
        Save evaluation results to file
        """
        self.results_df.to_csv(os.path.join(self.experiment_folder, "results_clipscore.csv"), index=False)
        
if __name__ == "__main__":

    argparser = argparse.ArgumentParser()
    argparser.add_argument("--experiment_name", type=str, default="default-experiment", help="Name of experiment to evaluate")
    argparser.add_argument("--evaluation_method", type=str, default="clipscore", help="Evaluation method to use", choices=["clipscore"])
    kwargs = vars(argparser.parse_args())

    if kwargs["evaluation_method"] == "clipscore":
        evaluation = CLIPScore(**kwargs)
    else:
        raise ValueError(f"Unknown evaluation method {kwargs['evaluation_method']}")
    
    evaluation.evaluate()
    evaluation.save_results()