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
        self.experiment_name = kwargs.get("experiment_name", "default-experiment")
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
        raise NotImplementedError
        
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
        self.result_dict = {
                "prompt_id": [],
                "image_id": [],
                "score": [],
                "user_prompt": [],
                "optimized_prompt": [],
                "image_path": [],
            }
        
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

            with torch.no_grad():
                # Tokenize and encode user prompt
                user_prompt = self.tokenizer([prompts[0]]).to(device)
                user_prompt_features = self.model.encode_text(user_prompt)
                user_prompt_features /= user_prompt_features.norm(dim=-1, keepdim=True)

                # load and encode generated images
                raw_images = [Image.open(os.path.join(prompt_folder, f"image_{image_idx}.png")) for image_idx in range(len(prompts))]
                images = torch.stack([self.preprocess(image) for image in raw_images]).to(device)
                images_features = self.model.encode_image(images)
                images_features /= images_features.norm(dim=-1, keepdim=True)

                # calculate CLIP-based similarity score
                scores = (100.0 * images_features @ user_prompt_features.T).data.cpu().squeeze(-1).numpy().tolist()

            # delete some objects due to OOM issues
            del user_prompt, user_prompt_features, images, images_features, raw_images

            # save results to global dict
            self.result_dict["prompt_id"].extend([int(prompt_folder.split("/")[-1])] * len(prompts))
            self.result_dict["image_id"].extend(list(range(len(prompts))))
            self.result_dict["score"].extend(scores)
            self.result_dict["user_prompt"].extend([prompts[0]] * len(prompts))
            self.result_dict["optimized_prompt"].extend(prompts)
            self.result_dict["image_path"].extend([os.path.join(prompt_folder, f"image_{image_idx}.png") for image_idx in range(len(prompts))])
            
            # # save results to dataframe
            # df_row = {
            #     "prompt_id": [int(prompt_folder.split("/")[-1])] * len(prompts),
            #     "image_id": list(range(len(prompts))),
            #     "score": scores,
            #     "user_prompt": [prompts[0]] * len(prompts),
            #     "optimized_prompt": prompts,
            #     "image_path": [os.path.join(prompt_folder, f"image_{image_idx}.png") for image_idx in range(len(prompts))],
            # }
            # self.results_df = pd.concat([self.results_df, pd.DataFrame.from_dict(df_row)])

    def save_results(self):
        """
        Save evaluation results to file
        """
        self.results_df = pd.DataFrame.from_dict(self.result_dict)
        self.results_df.to_csv(os.path.join(self.experiment_folder, f"results_clipscore_{self.experiment_name}.tsv"), index=False, sep="\t")
        
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