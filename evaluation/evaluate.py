import os, sys
import json
import argparse
from tqdm import tqdm

import pandas as pd
import torch
from torchvision import models, transforms
from PIL import Image
import open_clip

import pandas as pd

from pycocoevalcap.spice.spice import Spice

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

    def load_captions(self, folder: str):
        """
        Load prompts from given prompt-specific folder containing generated images
        """
        with open(os.path.join(folder, "captions.csv"), "r") as f:
            captions = f.read().splitlines()
            formatted_captions = []
            for caption in captions:
                formatted_captions.append(caption.split("\t")[-1])

        return formatted_captions

    def terminated_and_best_image(self, folder: str):
        """
        Load step at which loop was terminated and best image according to LLM
        """
        with open(os.path.join(folder, "results.txt"), "r") as f:
            lines = f.read().splitlines()
            for line in lines:
                if line.startswith("terminated at iteration"):
                    terminated = line.split("\t")[-1]
                if line.startswith("best image"):
                    best_image = line.split("\t")[-1]

        return terminated, best_image

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

    def return_df(self):
        return self.results_df

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

        self.results_df = pd.DataFrame(columns=["prompt_id", "image_id", "clip_score", "user_prompt", "optimized_prompt", "caption", "image_path"], index=["prompt_id", "image_id"])
        self.result_dict = {
                "prompt_id": [],
                "image_id": [],
                "clip_score": [],
                "user_prompt": [],
                "optimized_prompt": [],
                "caption": [],
                "image_path": [],
            }

    def encode_images(self, images):
        image = torch.stack([self.preprocess(image) for image in images]).to(device)
        images_features = self.model.encode_image(image)
        images_features /= images_features.norm(dim=-1, keepdim=True)
        return images_features
        
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
            captions = self.load_captions(prompt_folder)
            if len(captions) == len(prompts) - 1:
                #TODO temporary fix for missing captions; main bug is already fixed
                captions.append("")

            with torch.no_grad():
                # Tokenize and encode user prompt
                user_prompt = self.tokenizer([prompts[0]]).to(device)
                user_prompt_features = self.model.encode_text(user_prompt)
                user_prompt_features /= user_prompt_features.norm(dim=-1, keepdim=True)

                # load and encode generated images
                raw_images = [Image.open(os.path.join(prompt_folder, f"image_{image_idx}.png")) for image_idx in range(len(prompts))]
                images_features = self.encode_images(raw_images)

                # calculate CLIP-based similarity score
                #scores = (100.0 * images_features @ user_prompt_features.T).data.cpu().squeeze(-1).numpy().tolist()
                #features = torch.stack([torch.cosine_similarity(user_prompt_features, features) for features in images_features])
                features = torch.cosine_similarity(user_prompt_features, images_features, dim=-1)
                scores = (2.5 * torch.max(torch.zeros(len(images_features)), torch.cosine_similarity(user_prompt_features, images_features, dim=-1).data.cpu()).numpy()).tolist()

            # delete some objects due to OOM issues
            del user_prompt, user_prompt_features, images_features, raw_images

            # save results to global dict
            self.result_dict["prompt_id"].extend([int(prompt_folder.split("/")[-1])] * len(prompts))
            self.result_dict["image_id"].extend(list(range(len(prompts))))
            self.result_dict["clip_score"].extend(scores)
            self.result_dict["user_prompt"].extend([prompts[0]] * len(prompts))
            self.result_dict["optimized_prompt"].extend(prompts)
            self.result_dict["caption"].extend(captions)
            self.result_dict["image_path"].extend([os.path.join(prompt_folder, f"image_{image_idx}.png") for image_idx in range(len(prompts))])

    def save_results(self):
        """
        Save evaluation results to file
        """
        self.results_df = pd.DataFrame.from_dict(self.result_dict)
        self.results_df.to_csv(os.path.join(self.experiment_folder, f"results_clipscore_{self.experiment_name.split('/')[-1]}.tsv"), index=False, sep="\t")

class ImageSimilarity(Evaluate):
    """
    Evaluate how similar the original image is to the first and final generated images.
    """
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

        self.encode_images = CLIPScore(**kwargs).encode_images
        self.cos = torch.nn.CosineSimilarity(dim=0)
        self.results_df = pd.DataFrame(columns=["prompt_id", "image_id", "img_sim_score", "user_prompt", "optimized_prompt", "caption", "image_path"], index=["prompt_id", "image_id"])
        self.result_dict = {
                "prompt_id": [],
                "image_id": [],
                "img_sim_score": [],
                "user_prompt": [],
                "optimized_prompt": [],
                "caption": [],
                "image_path": [],
            }

    def cos_similarity(self, features_1, features_2):
        return self.cos(features_1.flatten(), features_2.flatten())

    def evaluate(self):
        """
        Evaluate similarity between the original image and the first generated image and the
        similarity between the original image and the final generated image.
        Their features are extracted with ResNet and compared with a cosine similarity method.

        The lower the score, the more similar the images are.
        """

        # iterate over all prompts/generations of experiment
        for prompt_folder in tqdm(os.listdir(self.experiment_folder)):
            prompt_folder = os.path.join(self.experiment_folder, prompt_folder)
            if not os.path.isdir(prompt_folder):
                continue
            prompts = self.load_prompts(prompt_folder)
            captions = self.load_captions(prompt_folder)
            if len(captions) == len(prompts) - 1:
                #TODO temporary fix for missing captions; main bug is already fixed
                captions.append("")


            with torch.no_grad():
                # load and encode generated images
                original_image = Image.open(os.path.join(prompt_folder, "original_image.png"))
                images = [Image.open(os.path.join(prompt_folder, f"image_{image_idx}.png")) for image_idx in range(len(prompts))]
                original_image_features = self.encode_images([original_image])
                images_features = self.encode_images(images)

                # calculate image similarity
                scores = torch.stack([self.cos_similarity(original_image_features, features) for features in images_features])
                scores = scores.tolist()
            # delete some objects due to OOM issues
            del images, images_features, original_image, original_image_features

            # save results to global dict
            self.result_dict["prompt_id"].extend([int(prompt_folder.split("/")[-1])] * len(prompts))
            self.result_dict["image_id"].extend(list(range(len(prompts))))
            self.result_dict["img_sim_score"].extend(scores)
            self.result_dict["user_prompt"].extend([prompts[0]] * len(prompts))
            self.result_dict["optimized_prompt"].extend(prompts)
            self.result_dict["caption"].extend(captions)
            self.result_dict["image_path"].extend([os.path.join(prompt_folder, f"image_{image_idx}.png") for image_idx in range(len(prompts))])

    def save_results(self):
        """
        Save evaluation results to file.
        """
        self.results_df = pd.DataFrame.from_dict(self.result_dict)
        self.results_df.to_csv(os.path.join(self.experiment_folder, f"results_image_similarity_{self.experiment_name.split('/')[-1]}.tsv"), index=False, sep="\t")

class CaptionEvaluation(Evaluate):
    """
    Evaluate generated images using CLIPScore-based metrics
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

        # Initialize scorers
        self.scorer = Spice()

        self.results_df = pd.DataFrame(
            columns=["prompt_id", "image_id", "spice_score", "user_prompt", "optimized_prompt", "caption", "image_path"],
            index=["prompt_id", "image_id"])
        self.result_dict = {
            "prompt_id": [],
            "image_id": [],
            "spice_score": [],
            "user_prompt": [],
            "optimized_prompt": [],
            "caption": [],
            "image_path": [],
        }

    def evaluate(self):
        """
        Evaluate captions using COCO captions metrics
        """
        # iterate over all prompts/generations of experiment
        for prompt_folder in tqdm(os.listdir(self.experiment_folder)):
            prompt_folder = os.path.join(self.experiment_folder, prompt_folder)
            if not os.path.isdir(prompt_folder):
                continue
            prompts = self.load_prompts(prompt_folder)
            captions = self.load_captions(prompt_folder)
            if len(captions) == len(prompts) - 1:
                #TODO temporary fix for missing captions; main bug is already fixed
                captions.append("")



            original_prompt = prompts[0]

            scores = []

            # calculate caption similarity
            for idx, caption in enumerate(captions):
                gts = {idx: [original_prompt]}
                res = {idx: [caption]}

                avg_score, _ = self.scorer.compute_score(gts, res)
                scores.append(avg_score)

            # save results to global dict
            self.result_dict["prompt_id"].extend([int(prompt_folder.split("/")[-1])] * len(prompts))
            self.result_dict["image_id"].extend(list(range(len(prompts))))
            self.result_dict["spice_score"].extend(scores)
            self.result_dict["user_prompt"].extend([prompts[0]] * len(prompts))
            self.result_dict["optimized_prompt"].extend(prompts)
            self.result_dict["caption"].extend(captions)
            self.result_dict["image_path"].extend(
                [os.path.join(prompt_folder, f"image_{image_idx}.png") for image_idx in range(len(prompts))])

    def save_results(self):
        """
        Save evaluation results to file
        """
        self.results_df = pd.DataFrame.from_dict(self.result_dict)
        self.results_df.to_csv(os.path.join(self.experiment_folder, f"results_caption_score_{self.experiment_name.split('/')[-1]}.tsv"),
                               index=False, sep="\t")


class LLMEvaluation(Evaluate):
    """
    Evaluation based on LLM
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

        # Initialize scorers
        self.results_df = pd.DataFrame(
            columns=["prompt_id", "image_id", "terminated", "best_image", "user_prompt", "optimized_prompt", "caption", "image_path"],
            index=["prompt_id", "image_id"])
        self.result_dict = {
            "prompt_id": [],
            "image_id": [],
            "terminated": [],
            "best_image": [],
            "user_prompt": [],
            "optimized_prompt": [],
            "caption": [],
            "image_path": [],
        }

    def evaluate(self):
        """
        Evaluate captions using COCO captions metrics
        """
        # iterate over all prompts/generations of experiment
        for prompt_folder in tqdm(os.listdir(self.experiment_folder)):
            prompt_folder = os.path.join(self.experiment_folder, prompt_folder)
            if not os.path.isdir(prompt_folder):
                continue
            prompts = self.load_prompts(prompt_folder)
            captions = self.load_captions(prompt_folder)
            if len(captions) == len(prompts) - 1:
                #TODO temporary fix for missing captions; main bug is already fixed
                captions.append("")
            terminated_at, best_image_num = self.terminated_and_best_image(prompt_folder)

            original_prompt = prompts[0]

            terminated = []
            best_image = []
            # calculate caption similarity
            for idx in range(len(captions)):
                if idx == int(terminated_at):
                    terminated.append(1)
                else:
                    terminated.append(0)
                if idx == int(best_image_num):
                    best_image.append(1)
                else:
                    best_image.append(0)


            # save results to global dict
            self.result_dict["prompt_id"].extend([int(prompt_folder.split("/")[-1])] * len(prompts))
            self.result_dict["image_id"].extend(list(range(len(prompts))))
            self.result_dict["terminated"].extend(terminated)
            self.result_dict["best_image"].extend(best_image)
            self.result_dict["user_prompt"].extend([prompts[0]] * len(prompts))
            self.result_dict["optimized_prompt"].extend(prompts)
            self.result_dict["caption"].extend(captions)
            self.result_dict["image_path"].extend(
                [os.path.join(prompt_folder, f"image_{image_idx}.png") for image_idx in range(len(prompts))])

    def save_results(self):
        """
        Save evaluation results to file
        """
        self.results_df = pd.DataFrame.from_dict(self.result_dict)
        self.results_df.to_csv(os.path.join(self.experiment_folder, f"llm_eval_{self.experiment_name}.tsv"),
                               index=False, sep="\t")


if __name__ == "__main__":

    argparser = argparse.ArgumentParser()
    argparser.add_argument("--experiment_name", type=str, default="default-experiment",
                           help="Name of experiment to evaluate")
    argparser.add_argument("--evaluation_method", type=str, default="clipscore", help="Evaluation method to use",
                           choices=["clipscore", "image_similarity", "caption_score", "llm_eval", "all"])
    kwargs = vars(argparser.parse_args())

    if kwargs["evaluation_method"] == "all":

        # Initialize all evaluations
        clip_eval = CLIPScore(**kwargs)
        img_sim_eval = ImageSimilarity(**kwargs)
        caption_eval = CaptionEvaluation(**kwargs)
        llm_eval = LLMEvaluation(**kwargs)

        if os.path.isfile(os.path.join(os.getcwd(), f'data/results/{kwargs["experiment_name"]}/000000/original_image.png')):
            evaluations = [clip_eval, caption_eval, llm_eval]
        else:
            evaluations = [clip_eval, img_sim_eval, caption_eval, llm_eval]

        # Define common columns for all evaluation results
        joiner = ["prompt_id", "image_id", "user_prompt", "optimized_prompt", "caption", "image_path"]

        # Combine the results of all evaluations
        for idx, evaluations in enumerate(evaluations):
            evaluations.evaluate()
            evaluations.save_results()
            results = evaluations.return_df()
            if idx == 0:
                all_results = results.copy()
            else:
                all_results = pd.merge(all_results, results, on=joiner, how='left')
        # Save the results to csv
        all_results.to_csv(os.path.join(os.getcwd(), f'data/results/{kwargs["experiment_name"]}', f'evaluation.tsv'),
                               index=False, sep="\t")

    else:
        if kwargs["evaluation_method"] == "clipscore":
            evaluation = CLIPScore(**kwargs)
        elif kwargs["evaluation_method"] == "image_similarity":
            evaluation = ImageSimilarity(**kwargs)
        elif kwargs["evaluation_method"] == "caption_score":
            evaluation = CaptionEvaluation(**kwargs)
        elif kwargs["evaluation_method"] == "llm_eval":
            evaluation = LLMEvaluation(**kwargs)
        else:
            raise ValueError(f"Unknown evaluation method {kwargs['evaluation_method']}")

        evaluation.evaluate()
        evaluation.save_results()
        test = evaluation.return_df()
