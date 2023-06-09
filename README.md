# SocraticImageGeneration
Optimizing image generation using the notion of Socratic Models

The advent of recent generative image models has allowed any user to creatively express themselves using arbitrary
textual prompts. However, it is often required to do extensive trial-and-error-based prompt engineering to obtain high-quality generations that are aligned
with the user intent. We propose an iterative approach of a pipeline that given a
prompt it generates an improved image by iteratively refining the prompt, from
the history of generated images and their refined captions for that prompt.

Demo notebook is available [here](blog/Socratic_Image_Generation_Showcase.ipynb)

## Installation

To create the Conda environment with all the necessary libraries 
```shell
conda env create -f environment.yaml
```

To activate the created environment:
```shell
conda activate SIG
```


[//]: # (Folder structure &#40;are all necessary folders created when cloning the repository&#41;)




## Running the Pipeline
### Important
To use the pipeline an OpenAI API Key is **required**, please add that in the config/openai_api_key.txt file, or 
provide the path using language_model --api_key_path argument (the key should be the only value in that file).


### Constructing the necessary subsets of COCO dataset 
For the experiments we used subsets of COCO dataset. To create the same subsets (and be able to run the experiments) use
this [Notebook](data/datasets/COCO_Captions_Data_Subsampling.ipynb).

## Reproducing the results
To reproduce the results that we present in the blogpost you need to run the following 2 commands:
(Keep in mind that the models are not returning always the same results, prompt and images, thus the reproduced
results might be different from the original)

> ```python -m model.run_model pipeline --mode full_experiment --experiment_name full_experiment_coco dataset --dataset cococaption-medium ```

> ```python -m model.run_model pipeline --mode full_experiment --experiment_name full_experiment_parti dataset --dataset parti-prompts-medium```

Note: You can also use the provided run_model.job file.
### Executing Pipeline for a Dataset

You can run the following command to execute the Pipeline for a dataset. More options for the experiments can be found 
below. The default values of the experiments can be found be providing the '--help' parameter to  the run_model module. 
> ```python -m model.run_model pipeline --mode full_experiment --experiment_name DEMO_EXPERIMENT ```

### Executing Pipeline for a Single Prompt
Change the {prompt_to_use} argument with your prompt.
> ``` python -m model.run_model pipeline --experiment_name DEMO_EXPERIMENT dataset --prompt '{prompt_to_use}'```

## Options for the experiments 

Group of arguments:  

**pipeline**  
>   --mode can be : 'inference' or 'full_experiment'\
    --experiment_name : String, Give a name for the experiment, the output files will be saved in a directory called with the experiment name\
    --max_cycles : Integer, Maximum number of times to optimize prompt and generate image \
    --terminate_on_similarity : Boolean, Whether to terminate the generation process when the language model regards the generated image and the original prompt as similar enough\
    --select_best_image : Boolean, Whether to select the best image from the generated images

**dataset**
> --dataset : String, The dataset to be used (available choices are shown below) \
> --prompt : String, Give a prompt just to run the pipeline only with that prompt.

**image_generator**
> --model : String, Image generator model (available choices are shown below) \
> --device_map' : String,  Device to run image generator on. \
> --torch_dtype : torch.dtype, Model Precision \
> --seed : integer, The seed for the model. \
> --num_inference_steps : integer, Number of inference steps. The more the better results but more execution time. \
> --guidance_scale : float, Classifier free guidance forces the generation to better match with the prompt \
> --height: integer, Height of the image to be generated."
> --width: integer, Width of the image to be generated."

**image_captioning**
> --device_map : String, Device to run image generator on. \
    --model : String, Image captioning model (available choices are shown below)'

**language_model**
>   --api_key_path: String, Path to text file containing api key \
    --template : String, Path to template to use for language model \
    --system_prompt : String, Path to system prompt to use for language model \
    --similarity_template : String, Path to template to use for similarity check \
    --system_sim_prompt : String, Path to system prompt to use for similarity check \
    --best_image_template : String, Path to template to use for best image selection \
    --system_best_image_prompt : String, Path to system prompt to use for best image selection
 
**Available Datasets to use:**
>   PartiPrompts = "parti-prompts" \
    PartiPromptsSmall = "parti-prompts-small" \
    PartiPromptsMedium = "parti-prompts-medium"\
    Flickr30k = "flickr30k" \
    Flickr30kSmall = "flickr30k-small" \
    CocoCaptionSmall = "cococaption-small" \
    CocoCaptionMedium = "cococaption-medium" \
    CocoCaptionLarge = "cococaption-large" 

**Available Image Generator models to use:**
>   StableDiffusionV1_4 = "CompVis/stable-diffusion-v1-4" \
    StableDiffusionV1_5 = "runwayml/stable-diffusion-v1-5" \
    StableDiffusionV2_1_Base = "stabilityai/stable-diffusion-2-1-base" \
    StableDiffusionV2_1 ="stabilityai/stable-diffusion-2-1"

**Available Image Captioning models to use:**
>  BLIP_LARGE = "blip_large"\
  BLIP2 = "blip_2"



## Evaluation

You can use the following command to evaluate the results that you produce by running the pipeline. 
(change the DEMO_EXP_EVAL with the name that you gave to the experiment)
### Generating Scores
> ```python -m evaluation.evaluate --experiment_name DEMO_EXPERIMENT --evaluation_method all ```

Furthermore, you can use the [Evaluation Notebook](evaluation/evalution_results.ipynb) to visualize the results.


**Available evaluation methods:**
>"clipscore", "image_similarity", "caption_score", "llm_eval", "all"

### Interpreting Evaluation scores
For better understanding and visualization of the evaluation scores we created a notebook with visualization of the results.
You can see tables and charts for each score individually and combined if you have scores from all the evaluation methods.
The notebook can be found in /evaluation/evaluation_results.ipynb