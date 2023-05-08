from tkinter import *
import tkinter as tk
from PIL import ImageTk, Image
import pandas as pd
import os
import glob

# Specify path to the results folder

path = input("Enter the path to the folder with results: ")

# Create windom
root = tk.Tk()
root.resizable()
# root.title("Provide path to the results folder")

# Frame to display images
frame = tk.Frame(root)
frame.pack(pady=10)

# Specify path to the images


folders = sorted([x[0] for x in os.walk(path)][1:])
i = 0
folder = folders[i]

root.title(f"Results viewer: {folder.rsplit('/', 1)[-1]}")

captions = pd.read_csv(f'{folder}/captions.csv', sep='\t', header=None)
captions = captions[1].tolist()
captions.append('')
for i, caption in enumerate(captions):
    if i != 5:
        captions[i] = 'Caption: ' + caption

prompts = pd.read_csv(f'{folder}/prompts.csv', sep='\t', header=None)
prompts = prompts[1].tolist()
for i, prompt in enumerate(prompts):
    prompts[i] = 'Prompt: ' + prompt

List_img = []
for image in sorted(glob.glob(folder + '/*.png')):
    List_img.append(ImageTk.PhotoImage(Image.open(image)))


#creating label to display images and captions
j = 0
img_label = Label(frame, image=List_img[j])
img_label.pack()
prompt_label = Label(frame, text=prompts[j])
prompt_label.pack()
caption_label = Label(frame, text=captions[j])
caption_label.pack()
root.title(f"Results viewer: {folder.rsplit('/', 1)[-1]} (Original image)")


def up(event):
    global i
    i = i + 1
    if i >= len(folders):
        i = 0
    global folder
    folder = folders[i]

    # Load captions
    global captions
    captions = pd.read_csv(f'{folder}/captions.csv', sep='\t',
                           header=None)
    captions = captions[1].tolist()
    captions.append('')
    for idx, caption in enumerate(captions):
        if idx != 5:
            captions[idx] = 'Caption: ' + caption

    # Load prompts
    global prompts
    prompts = pd.read_csv(f'{folder}/prompts.csv', sep='\t', header=None)
    prompts = prompts[1].tolist()
    for idx, prompt in enumerate(prompts):
        prompts[idx] = 'Prompt: ' + prompt

    # Load images
    global List_img
    List_img = []
    for image in sorted(glob.glob(folder + '/*.png')):
        List_img.append(ImageTk.PhotoImage(Image.open(image)))

    global j
    j = 0

    root.title(f"Results viewer: {folder.rsplit('/', 1)[-1]} (Original image)")

    prompt_label.config(text=prompts[j])
    img_label.config(image=List_img[j])
    caption_label.config(text=captions[j])

def down(event):
    global i
    i = i - 1
    if i < 0:
        i = len(folders) - 1

    global folder
    folder = folders[i]

    root.title(f"Results viewer: {folder.rsplit('/', 1)[-1]}\n Original image")


    # Load captions
    global captions
    captions = pd.read_csv(f'{folder}/captions.csv', sep='\t',
                           header=None)
    captions = captions[1].tolist()
    captions.append('')
    for idx, caption in enumerate(captions):
        if idx != 5:
            captions[idx] = 'Caption: ' + caption

    # Load prompts
    global prompts
    prompts = pd.read_csv(f'{folder}/prompts.csv', sep='\t', header=None)
    prompts = prompts[1].tolist()
    for idx, prompt in enumerate(prompts):
        prompts[idx] = 'Prompt: ' + prompt

    # Load images
    global List_img
    List_img = []
    for image in sorted(glob.glob(folder + '/*.png')):
        List_img.append(ImageTk.PhotoImage(Image.open(image)))

    global j
    j = 0

    root.title(f"Results viewer: {folder.rsplit('/', 1)[-1]} (Original image)")

    prompt_label.config(text=prompts[j])
    img_label.config(image=List_img[j])
    caption_label.config(text=captions[j])


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

   prompt_label.config(text=prompts[j])
   img_label.config(image=List_img[j])
   caption_label.config(text=captions[j])

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

   prompt_label.config(text=prompts[j])
   img_label.config(image=List_img[j])
   caption_label.config(text=captions[j])

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


# def specify_path():
#     global path
#     path = user_path.get()
#     run()
#
#
#
#
# user_path = tk.Entry(root)
# user_path.pack()
#
# tk.Button(
#     root,
#     text="Submit",
#     padx=10,
#     pady=5,
#     command=specify_path
# ).pack()
#
# root.mainloop()