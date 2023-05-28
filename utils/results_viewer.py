from tkinter import *
import tkinter as tk
from PIL import ImageTk, Image
import pandas as pd
import os
import glob

# Specify path to the results folder

path = input("Enter the path to the folder with results: ")
evaluation_file = path + '/evaluation.tsv'

# scores
results = pd.read_csv(evaluation_file, sep='\t')

dataset = input("What dataset you want to use (coco, parti): ")

if dataset == 'coco':
    scores_names = ['clip_score','spice_score','img_sim_score']
elif dataset == 'parti':
    scores_names = ['clip_score','spice_score']
else:
    print('Wrong dataset name')
    exit()

scores = results[['prompt_id', 'image_id'] + scores_names]


# Create windom
root = tk.Tk()
root.resizable()
# root.title("Provide path to the results folder")

# Frame to display images
frame = tk.Frame(root)
frame.pack(pady=10)

# Specify path to the images

# path = '/Users/slawek/PycharmProjects/SocraticImageGeneration/data/results/full_experiment_V1_coco'

folders = sorted([x[0] for x in os.walk(path)][1:])
i = 0
folder = folders[i]

root.title(f"Results viewer: {folder.rsplit('/', 1)[-1]}")

prompts = pd.read_csv(f'{folder}/prompts.csv', sep='\t', header=None)
prompts = prompts[1].tolist()
if dataset == 'coco':
    prompts.insert(0, 'Original image')


captions = pd.read_csv(f'{folder}/captions.csv', sep='\t', header=None)
captions = captions[1].tolist()
if dataset == 'coco':
    captions.insert(0, prompts[1])
for i, caption in enumerate(captions):
    captions[i] = 'Caption: ' + caption

for i, prompt in enumerate(prompts):
    prompts[i] = 'Prompt: ' + prompt

List_img = []
if dataset == 'coco':
    image_paths = glob.glob(folder + '/original_image.png') + sorted(glob.glob(folder + '/image_*.png'))
else:
    image_paths = sorted(glob.glob(folder + '/image_*.png'))
for image in image_paths:
    List_img.append(ImageTk.PhotoImage(Image.open(image)))


#creating label to display images and captions
j = 0
scores_to_show = scores[(scores['prompt_id'] == j) & (scores['image_id'] == j)][scores_names].values

img_label = Label(frame, image=List_img[j])
img_label.pack()
prompt_label = Label(frame, text=prompts[j])
prompt_label.pack()
caption_label = Label(frame, text=captions[j])
caption_label.pack()
scores_label = Label(frame, text="")
scores_label.pack()


def up(event):
    global i
    i = i + 1
    if i >= len(folders):
        i = 0
    global folder
    folder = folders[i]

    # Load prompts
    global prompts
    prompts = pd.read_csv(f'{folder}/prompts.csv', sep='\t', header=None)
    prompts = prompts[1].tolist()
    if dataset == 'coco':
        prompts.insert(0, 'Original image')

    # Load captions
    global captions
    captions = pd.read_csv(f'{folder}/captions.csv', sep='\t',
                           header=None)
    captions = captions[1].tolist()
    if dataset == 'coco':
        captions.insert(0, prompts[1])
    for idx, caption in enumerate(captions):
        captions[idx] = 'Caption: ' + caption

    for idx, prompt in enumerate(prompts):
        prompts[idx] = 'Prompt: ' + prompt


    # Load images
    global List_img
    List_img = []
    if dataset == 'coco':
        image_paths = glob.glob(folder + '/original_image.png') + sorted(glob.glob(folder + '/image_*.png'))
    else:
        image_paths = sorted(glob.glob(folder + '/image_*.png'))
    for image in image_paths:
        List_img.append(ImageTk.PhotoImage(Image.open(image)))

    global j
    j = 0

    global scores_to_show
    if dataset == 'coco':
        if j == 0:
            prompt_label.config(text=prompts[j])
            img_label.config(image=List_img[j])
            caption_label.config(text=captions[j])
            scores_label.config(text="")
        else:
            prompt_label.config(text=prompts[j])
            img_label.config(image=List_img[j])
            caption_label.config(text=captions[j])
            scores_to_show = scores[(scores['prompt_id'] == i) & (scores['image_id'] == j-1)][scores_names].values
            scores_label.config(
                text=f'Clip score: {round(scores_to_show[0][0], 3)}\nSpice score: {round(scores_to_show[0][1], 3)}\nImage similarity score: {round(scores_to_show[0][2], 3)}')
        root.title(f"Results viewer: {folder.rsplit('/', 1)[-1]} (Original image)")
    else:
        prompt_label.config(text=prompts[j])
        img_label.config(image=List_img[j])
        caption_label.config(text=captions[j])
        scores_to_show = scores[(scores['prompt_id'] == i) & (scores['image_id'] == j)][scores_names].values
        scores_label.config(
            text=f'Clip score: {round(scores_to_show[0][0], 3)}\nSpice score: {round(scores_to_show[0][1], 3)}')
        root.title(f"Results viewer: {folder.rsplit('/', 1)[-1]}")

def down(event):
    global i
    i = i - 1
    if i < 0:
        i = len(folders) - 1

    global folder
    folder = folders[i]

    root.title(f"Results viewer: {folder.rsplit('/', 1)[-1]}\n Original image")

    # Load prompts
    global prompts
    prompts = pd.read_csv(f'{folder}/prompts.csv', sep='\t', header=None)
    prompts = prompts[1].tolist()
    if dataset == 'coco':
        prompts.insert(0, 'Original image')

    # Load captions
    global captions
    captions = pd.read_csv(f'{folder}/captions.csv', sep='\t',
                           header=None)
    captions = captions[1].tolist()
    if dataset == 'coco':
        captions.insert(0, prompts[1])
    for idx, caption in enumerate(captions):
        captions[idx] = 'Caption: ' + caption

    for idx, prompt in enumerate(prompts):
        prompts[idx] = 'Prompt: ' + prompt

    # Load images
    global List_img
    List_img = []
    if dataset == 'coco':
        image_paths = glob.glob(folder + '/original_image.png') + sorted(glob.glob(folder + '/image_*.png'))
    else:
        image_paths = sorted(glob.glob(folder + '/image_*.png'))
    for image in image_paths:
        List_img.append(ImageTk.PhotoImage(Image.open(image)))

    global j
    j = 0

    global scores_to_show
    if dataset == 'coco':
        if j == 0:
            prompt_label.config(text=prompts[j])
            img_label.config(image=List_img[j])
            caption_label.config(text=captions[j])
            scores_label.config(text="")
        else:
            prompt_label.config(text=prompts[j])
            img_label.config(image=List_img[j])
            caption_label.config(text=captions[j])
            scores_to_show = scores[(scores['prompt_id'] == i) & (scores['image_id'] == j-1)][scores_names].values
            scores_label.config(
                text=f'Clip score: {round(scores_to_show[0][0], 3)}\nSpice score: {round(scores_to_show[0][1], 3)}\nImage similarity score: {round(scores_to_show[0][2], 3)}')
        root.title(f"Results viewer: {folder.rsplit('/', 1)[-1]} (Original image)")
    else:
        prompt_label.config(text=prompts[j])
        img_label.config(image=List_img[j])
        caption_label.config(text=captions[j])
        scores_to_show = scores[(scores['prompt_id'] == i) & (scores['image_id'] == j)][scores_names].values
        scores_label.config(
            text=f'Clip score: {round(scores_to_show[0][0], 3)}\nSpice score: {round(scores_to_show[0][1], 3)}')
        root.title(f"Results viewer: {folder.rsplit('/', 1)[-1]}")
#function for next image
def next_img(event):
    global j
    j = j + 1
    if j >= len(List_img):
       j = 0

    if j == 0:
       root.title(f"Results viewer: {folder.rsplit('/', 1)[-1]} (Original image)")
    else:
       root.title(f"Results viewer: {folder.rsplit('/', 1)[-1]} (Generated image {j})")

    global scores_to_show
    if dataset == 'coco':
       if j == 0:
           prompt_label.config(text=prompts[j])
           img_label.config(image=List_img[j])
           caption_label.config(text=captions[j])
           scores_label.config(text="")
       else:
           prompt_label.config(text=prompts[j])
           img_label.config(image=List_img[j])
           caption_label.config(text=captions[j])
           scores_to_show = scores[(scores['prompt_id'] == i) & (scores['image_id'] == j-1)][scores_names].values
           scores_label.config(
               text=f'Clip score: {round(scores_to_show[0][0], 3)}\nSpice score: {round(scores_to_show[0][1], 3)}\nImage similarity score: {round(scores_to_show[0][2], 3)}')

    else:
        prompt_label.config(text=prompts[j])
        img_label.config(image=List_img[j])
        caption_label.config(text=captions[j])
        scores_to_show = scores[(scores['prompt_id'] == i) & (scores['image_id'] == j)][scores_names].values
        scores_label.config(
            text=f'Clip score: {round(scores_to_show[0][0], 3)}\nSpice score: {round(scores_to_show[0][1], 3)}')

#function for prev image
def prev(event):
    global j
    j = j - 1
    if j < 0:
       j = len(List_img) - 1

    if j == 0:
       root.title(f"Results viewer: {folder.rsplit('/', 1)[-1]} (Original image)")
    else:
       root.title(f"Results viewer: {folder.rsplit('/', 1)[-1]} (Generated image {j})")

    global scores_to_show
    if dataset == 'coco':
       if j == 0:
           prompt_label.config(text=prompts[j])
           img_label.config(image=List_img[j])
           caption_label.config(text=captions[j])
           scores_label.config(text="")
       else:
           prompt_label.config(text=prompts[j])
           img_label.config(image=List_img[j])
           caption_label.config(text=captions[j])
           scores_to_show = scores[(scores['prompt_id'] == i) & (scores['image_id'] == j-1)][scores_names].values
           scores_label.config(
               text=f'Clip score: {round(scores_to_show[0][0], 3)}\nSpice score: {round(scores_to_show[0][1], 3)}\nImage similarity score: {round(scores_to_show[0][2], 3)}')
    else:
        prompt_label.config(text=prompts[j])
        img_label.config(image=List_img[j])
        caption_label.config(text=captions[j])
        scores_to_show = scores[(scores['prompt_id'] == i) & (scores['image_id'] == j)][scores_names].values
        scores_label.config(
            text=f'Clip score: {round(scores_to_show[0][0], 3)}\nSpice score: {round(scores_to_show[0][1], 3)}')

#creating frame for previous, next, and exit button
frame1 = tk.Frame(root)
frame1.pack(pady=5)
# Prev = tk.Button(frame1, text="Previous", command=prev)
# Prev.pack(side="left", padx=10)
root.bind('<Left>', prev)

# Next = tk.Button(frame1, text="Next", command=next_img)
# Next.pack(side="right", padx=10)
root.bind('<Right>', next_img)

# Up = tk.Button(frame1, text="Up", command=up)
# Up.pack(side="top", padx=10)
root.bind('<Up>', up)

# Down = tk.Button(frame1, text="Down", command=down)
# Down.pack(side="bottom", padx=10)
root.bind('<Down>', down)

root.mainloop()
