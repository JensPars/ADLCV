import os
import random
import gradio as gr

# Set the paths to your two directories of images
dir1_path = "path/to/directory1"
dir2_path = "path/to/directory2"

# Initialize the vote tallies for each directory
dir1_votes = 0
dir2_votes = 0

def load_image_paths(dir_path):
    return [os.path.join(dir_path, filename) for filename in os.listdir(dir_path)]

def update_votes(dir1_path, dir2_path, choice):
    global dir1_votes, dir2_votes
    if choice == "Image 1":
        dir1_votes += 1
    else:
        dir2_votes += 1
    
    # Save the updated vote tallies to disk
    with open("vote_tallies.txt", "w") as file:
        file.write(f"Directory 1: {dir1_votes}\n")
        file.write(f"Directory 2: {dir2_votes}\n")
    
    # Randomly select a new pair of images
    dir1_images = load_image_paths(dir1_path)
    dir2_images = load_image_paths(dir2_path)
    random_image1 = random.choice(dir1_images)
    random_image2 = random.choice(dir2_images)
    
    return random_image1, random_image2

# Load the initial pair of images
dir1_images = load_image_paths(dir1_path)
dir2_images = load_image_paths(dir2_path)
random_image1 = random.choice(dir1_images)
random_image2 = random.choice(dir2_images)

# Create the Gradio interface
iface = gr.Interface(
    fn=update_votes,
    inputs=[
        gr.Image(value=random_image1, label="Image 1"),
        gr.Image(value=random_image2, label="Image 2"),
        gr.Radio(["Image 1", "Image 2"], label="Vote")
    ],
    outputs=[
        gr.Image(label="Image 1"),
        gr.Image(label="Image 2")
    ],
    title="Image Voting App",
    description="Vote for the better image among the two presented.",
    allow_flagging=False
)

# Launch the Gradio app
iface.launch()