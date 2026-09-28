from flask import Flask, render_template, request, jsonify
import torch
from diffusers import AutoPipelineForText2Image
import io
import base64

app = Flask(__name__)

print("Loading Lightweight Image Generator Model (SD-Turbo)...")
# AutoPipelineForText2Image automatically correct model config mapping handle karta hai
image_pipe = AutoPipelineForText2Image.from_pretrained(
    "stabilityai/sd-turbo", 
    torch_dtype=torch.float32 if not torch.cuda.is_available() else torch.float16, 
    variant="fp16" if torch.cuda.is_available() else None
)
image_pipe.to("cuda" if torch.cuda.is_available() else "cpu")
print("Image Model Loaded Successfully!")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/generate", methods=["POST"])
def generate():
    data = request.json or {}
    user_prompt = data.get("prompt", "")

    if not user_prompt:
        return jsonify({"error": "Please enter an image prompt."}), 400

    try:
        # SD-Turbo ke liye guidance_scale=0.0 aur num_inference_steps=1 ya 4 best hain
        image = image_pipe(user_prompt, num_inference_steps=4, guidance_scale=0.0).images[0]
        
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

        return jsonify({"image": f"data:image/jpeg;base64,{img_str}"})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860)
