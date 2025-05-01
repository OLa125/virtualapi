# PromptDresser Flask API

This project provides an API interface for the PromptDresser virtual try-on model using Flask.

## 🔧 Installation

1. Clone the repository or unzip the project folder.
2. Install dependencies:
```bash
pip install -r requirements.txt
```
3. Download the pretrained models and place them under `pretrained_models/` directory, or update paths in `inference_api.py`.

## 🚀 Running the API

```bash
python app.py
```
This will start the server at `http://0.0.0.0:7860`

## 🧪 API Endpoint

### POST `/generate`
Generate a virtual try-on image based on a human and cloth image, along with an optional prompt.

#### Request (JSON):
```json
{
  "prompt": "a photo of a woman in a red dress",
  "cloth_image": "<base64_encoded_png>",
  "human_image": "<base64_encoded_png>"
}
```

#### Response (JSON):
```json
{
  "result": "<base64_encoded_output_image>"
}
```

## 📁 Directory Structure
```
project_root/
├── app.py
├── inference_api.py
├── requirements.txt
├── README.md
├── configs/
│   └── VITONHD.yaml
├── pretrained_models/
│   └── [downloaded models here]
└── promptdresser/
    └── [core project code]
```

## 📦 Model Sources
You can upload models to Hugging Face Hub or provide a direct link to download them if they are too large to include directly.

---
Developed for deployment of PromptDresser-based virtual try-on using minimal API interface.
