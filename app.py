# app.py

from flask import Flask, request, jsonify
from inference_api import PromptDresserPipeline
from PIL import Image
import io
import base64

app = Flask(__name__)

# Load model once when server starts
pipeline = PromptDresserPipeline(config_path="./configs/VITONHD.yaml")

@app.route("/generate", methods=["POST"])
def generate():
    try:
        # Expecting JSON with keys like 'prompt', 'cloth_image', 'human_image' (all base64 encoded)
        data = request.get_json()

        # Decode base64 images
        cloth_image = Image.open(io.BytesIO(base64.b64decode(data["cloth_image"]))).convert("RGB")
        human_image = Image.open(io.BytesIO(base64.b64decode(data["human_image"]))).convert("RGB")

        # Prepare inputs
        inputs = {
            "prompt": data.get("prompt", ""),
            "cloth_image": cloth_image,
            "human_image": human_image
        }

        # Run inference
        result_image = pipeline.generate(inputs)

        # Convert output image to base64
        buffered = io.BytesIO()
        result_image.save(buffered, format="PNG")
        result_b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

        return jsonify({"result": result_b64})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860)
