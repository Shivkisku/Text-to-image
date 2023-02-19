import tkinter as tk
import customtkinter as ctk 
from PIL import ImageTk
from authtoken import auth_token
import torch
from torch.cuda.amp import autocast
from diffusers import StableDiffusionPipeline 

# Create the app
app = tk.Tk()
app.geometry("652x782")
app.title("Stable Bud") 
ctk.set_appearance_mode("dark") 

# Prompt for text input
prompt = ctk.CTkEntry(height=40, width=512, text_font=("Arial", 20), text_color="black", fg_color="white") 
prompt.place(x=10, y=10)

# Label to display generated image
lmain = ctk.CTkLabel(height=512, width=512)
lmain.place(x=10, y=110)

# Load the model
modelid = "CompVis/stable-diffusion-v1-4"
device = "cuda" if torch.cuda.is_available() else "cpu"
pipe = StableDiffusionPipeline.from_pretrained(modelid, revision="fp16", torch_dtype=torch.float16, use_auth_token=auth_token) 
pipe.to(device) 

# Function to generate image and display it
def generate(): 
    with autocast(enabled=device == "cuda"): 
        image = pipe(prompt.get(), guidance_scale=8.5)["sample"][0]
    
    image.save('generatedimage.png')
    img = ImageTk.PhotoImage(image)
    lmain.configure(image=img) 

# Button to trigger image generation
trigger = ctk.CTkButton(height=40, width=120, text_font=("Arial", 20), text_color="white", fg_color="blue", command=generate) 
trigger.configure(text="Generate") 
trigger.place(x=206, y=60) 

app.mainloop()
