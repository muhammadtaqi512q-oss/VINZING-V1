from flask import Flask, render_template, request, jsonify
import torch
from transformers import pipeline
import io
import base64

app = Flask(__name__)

print("Loading Lightweight Image Generator Model (Under 5GB)...")
# SD-Turbo ek fast aur lightweight image generation model hai (approx 2-3 GB)
image_pipe = pipeline(
    "text-to-image",
    model="stabilityai/sd-turbo",
    torch_dtype=torch.float32 if not torch.cuda.is_available() else torch.float16,
    device_map="auto"
)
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
        # Prompt ke mutabiq image generate karna (SD-Turbo ke liye num_inference_steps=4 best hai)
        image = image_pipe(user_prompt, num_inference_steps=4, guidance_scale=0.0).images[0]
        
        # Image ko base64 format mein convert karna taaki frontend par direct show ho sake
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

        return jsonify({"image": f"data:image/jpeg;base64,{img_str}"})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860)
