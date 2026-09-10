import streamlit as st
import numpy as np
import os
import torch
import torch.nn.functional as F
import timm
import cv2
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

st.set_page_config(page_title="Skin Cancer Detection", layout="centered")

device = torch.device("cpu")

# -----------------------------
# Load Model (Local or Hugging Face Hub)
# -----------------------------

HF_REPO_ID = os.environ.get("HF_MODEL_REPO", "YashShinde11/skin-cancer-swin")
MODEL_FILENAME = "swin_model_new.pth"

def get_model_path():
    """Locate local model weights or automatically fetch from Hugging Face Hub."""
    # 1. Check local models/ directory (relative to repo root)
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    local_path = os.path.join(base_dir, "models", MODEL_FILENAME)
    if os.path.exists(local_path):
        return local_path

    # 2. Check current working directory / models
    cwd_path = os.path.join("models", MODEL_FILENAME)
    if os.path.exists(cwd_path):
        return cwd_path

    # 3. Streamlit Cloud / Production: Download from Hugging Face Hub
    try:
        from huggingface_hub import hf_hub_download
        with st.spinner("Fetching model weights from Hugging Face Hub... (one-time download)"):
            hf_path = hf_hub_download(
                repo_id=HF_REPO_ID,
                filename=MODEL_FILENAME
            )
            return hf_path
    except Exception as e:
        st.error(
            f"Could not load model locally and failed to download from Hugging Face ({HF_REPO_ID}).\n"
            f"Please ensure '{MODEL_FILENAME}' is uploaded to Hugging Face repository '{HF_REPO_ID}'.\n"
            f"Details: {e}"
        )
        raise e

@st.cache_resource
def load_model():
    model = timm.create_model(
        "swin_tiny_patch4_window7_224",
        pretrained=False,
        num_classes=7
    )
    model_path = get_model_path()
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint)
    model.eval()
    model.to(device)
    return model

model = load_model()

# -----------------------------
# Class Names
# -----------------------------

cancer_classes = [
    'actinic keratosis',
    'basal cell carcinoma',
    'melanoma',
    'nevus',
    'pigmented benign keratosis',
    'squamous cell carcinoma',
    'vascular lesion'
]

# -----------------------------
# Transform
# -----------------------------

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# -----------------------------
# GradCAM Class
# -----------------------------

class GradCAM:

    def __init__(self, model, target_layer):
        self.model = model
        self.gradients = None
        self.activations = None

        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output.detach()

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor, class_idx):
        self.model.zero_grad()

        output = self.model(input_tensor)
        loss = output[:, class_idx]
        loss.backward()

        gradients = self.gradients
        activations = self.activations

        # timm Swin uses (B, H, W, C) — permute to (B, C, H, W)
        if gradients.dim() == 4 and gradients.shape[-1] != gradients.shape[1]:
            gradients = gradients.permute(0, 3, 1, 2)
            activations = activations.permute(0, 3, 1, 2)

        weights = torch.mean(gradients, dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * activations, dim=1)
        cam = F.relu(cam)

        cam = cam.squeeze().cpu().numpy()
        cam = cv2.resize(cam, (224, 224))
        cam = cam - np.min(cam)
        cam = cam / (np.max(cam) + 1e-8)

        return cam

# -----------------------------
# GradCAM instance
# -----------------------------

gradcam = GradCAM(model, model.layers[-1].blocks[-1].norm1)

# -----------------------------
# Predict + GradCAM function
# -----------------------------

def predict_and_gradcam(image: Image.Image):

    image_resized = image.resize((224, 224))
    image_np = np.array(image_resized)

    # --- Prediction (no_grad) ---
    with torch.no_grad():
        input_tensor = transform(image).unsqueeze(0).to(device)
        output = model(input_tensor)
        probs = torch.softmax(output, dim=1)
        class_idx = torch.argmax(probs).item()
        confidence = probs[0][class_idx].item() * 100

    prediction = cancer_classes[class_idx]

    # --- GradCAM (needs grad) ---
    input_tensor_grad = transform(image).unsqueeze(0).to(device)
    input_tensor_grad.requires_grad_(True)

    cam = gradcam.generate(input_tensor_grad, class_idx)

    # --- Build overlay ---
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay = (heatmap_rgb * 0.4 + image_np * 0.6).astype(np.uint8)

    return prediction, confidence, image_np, heatmap_rgb, overlay

# -----------------------------
# Streamlit UI
# -----------------------------

st.markdown("## 🔬 Skin Cancer Detection with GradCAM")
st.markdown("Upload a dermoscopy image to get a prediction and visual explanation.")

uploaded = st.file_uploader("Upload a medical image", type=["jpg", "jpeg", "png"])

if uploaded:
    image = Image.open(uploaded).convert("RGB")

    st.image(image, caption="Uploaded Image", use_container_width=True)

    if st.button("🔮 Predict & Generate GradCAM"):

        with st.spinner("Analyzing image..."):
            prediction, confidence, image_np, heatmap_rgb, overlay = predict_and_gradcam(image)

        # --- Result ---
        st.success(f"**Predicted:** {prediction.title()}")
        st.info(f"**Confidence:** {confidence:.2f}%")

        # --- Confidence bar ---
        st.progress(int(confidence))

        st.markdown("---")
        st.markdown("### 🧠 GradCAM Visual Explanation")
        st.caption("The heatmap highlights regions the model focused on to make its prediction.")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.image(image_np, caption="Original", use_container_width=True)

        with col2:
            st.image(heatmap_rgb, caption="GradCAM Heatmap", use_container_width=True)

        with col3:
            st.image(overlay, caption="Overlay", use_container_width=True)

        # --- Warning for high-risk classes ---
        high_risk = ["melanoma", "basal cell carcinoma", "squamous cell carcinoma"]
        if prediction in high_risk:
            st.warning(
                f"⚠️ **{prediction.title()}** is a high-risk condition. "
                "Please consult a dermatologist immediately."
            )
        else:
            st.success("✅ No high-risk condition detected. Monitor and consult a doctor if concerned.")