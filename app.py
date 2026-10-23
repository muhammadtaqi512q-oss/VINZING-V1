from flask import Flask, render_template, request, jsonify
import torch
from diffusers import AutoPipelineForText2Image, AutoPipelineForImage2Image
import io
import base64
from PIL import Image

app = Flask(__name__)

print("Loading High-Quality Model (SDXL-Turbo)...")
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if torch.cuda.is_available() else torch.float32

# 1. New & Better Model: SDXL-Turbo (Text-to-Image)
image_pipe = AutoPipelineForText2Image.from_pretrained(
    "stabilityai/sdxl-turbo", 
    torch_dtype=dtype, 
    variant="fp16" if torch.cuda.is_available() else None
)
image_pipe.to(device)

# 2. Image-to-Image Pipeline (Zero Extra Download - Shares weights automatically)
print("Initializing Image-to-Image Pipeline...")
img2img_pipe = AutoPipelineForImage2Image.from_pipe(image_pipe)
print("Both Text-to-Image & Image-to-Image Models Loaded Successfully!")

@app.route("/")
def home():
    return render_template("index.html")

# Text-to-Image Route
@app.route("/generate", methods=["POST"])
def generate():
    data = request.json or {}
    user_prompt = data.get("prompt", "")

    if not user_prompt:
        return jsonify({"error": "Please enter an image prompt."}), 400

    try:
        # SDXL-Turbo ke liye 1 ya 2 steps aur guidance_scale=0.0 best hain
        image = image_pipe(user_prompt, num_inference_steps=2, guidance_scale=0.0).images[0]
        
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

        return jsonify({"image": f"data:image/jpeg;base64,{img_str}"})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Image-to-Image Route
@app.route("/generate-img2img", methods=["POST"])
def generate_img2img():
    data = request.json or {}
    user_prompt = data.get("prompt", "")
    init_image_b64 = data.get("image", "")

    if not user_prompt or not init_image_b64:
        return jsonify({"error": "Prompt and input image are required."}), 400

    try:
        if "," in init_image_b64:
            init_image_b64 = init_image_b64.split(",")[1]
        
        image_bytes = base64.b64decode(init_image_b64)
        init_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        
        # SDXL ke liye standard size 512x512 ya 1024x1024 (512x512 fast speed ke liye best hai)
        init_image = init_image.resize((512, 512))

        # Image-to-Image Execution
        image = img2img_pipe(
            prompt=user_prompt,
            image=init_image,
            strength=0.7,           
            num_inference_steps=2,
            guidance_scale=0.0
        ).images[0]
        
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

        return jsonify({"image": f"data:image/jpeg;base64,{img_str}"})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860)
