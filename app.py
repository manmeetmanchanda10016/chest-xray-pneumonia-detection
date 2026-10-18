
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import numpy as np
import cv2

st.set_page_config(
    page_title="Chest X-ray AI",
    page_icon="🩻",
    layout="wide"
)

device = torch.device("cpu")

st.markdown("""
<style>
.stApp {
    background: radial-gradient(circle at top left, #24104f 0%, #0b1020 45%, #080b14 100%);
    color: #f1f5f9;
}
.block-container {
    max-width: 1200px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}
header[data-testid="stHeader"] {
    background: transparent;
}
.hero {
    background: linear-gradient(120deg, #160b35, #31206b, #701a75);
    border: 1px solid #9333ea;
    padding: 35px;
    border-radius: 24px;
    text-align: center;
    margin-bottom: 25px;
    box-shadow: 0 0 30px #8b5cf633;
}
.hero h1 {
    color: #ffffff;
    font-size: 42px;
    margin-bottom: 8px;
}
.hero p {
    color: #c4b5fd;
    font-size: 17px;
}
.panel {
    background: linear-gradient(145deg, #151b30, #101426);
    border: 1px solid #37305c;
    padding: 25px;
    border-radius: 20px;
    box-shadow: 0 0 22px #7c3aed12;
    min-height: 250px;
}
h2, h3 {
    color: #e9d5ff !important;
}
p, label, .stMarkdown {
    color: #dbe4f0;
}
div[data-testid="stFileUploader"] {
    background: #101426;
    border: 1px dashed #8b5cf6;
    border-radius: 14px;
    padding: 12px;
}
div.stButton > button {
    background: linear-gradient(90deg, #7c3aed, #db2777, #0891b2);
    color: white;
    border: none;
    border-radius: 12px;
    height: 3.2em;
    font-weight: bold;
    width: 100%;
    box-shadow: 0 0 18px #9333ea45;
    transition: 0.2s;
}
div.stButton > button:hover {
    border: 1px solid #67e8f9;
    color: white;
    box-shadow: 0 0 25px #06b6d455;
}
.warning {
    background: #291b16;
    color: #fed7aa;
    padding: 16px;
    border-radius: 13px;
    border-left: 5px solid #fb923c;
    margin-bottom: 24px;
}
.result-normal {
    background: linear-gradient(135deg, #062e2b, #0c252d);
    border: 1px solid #2dd4bf;
    padding: 22px;
    border-radius: 17px;
    color: #99f6e4;
    box-shadow: 0 0 22px #14b8a622;
}
.result-pneumonia {
    background: linear-gradient(135deg, #3b1024, #301526);
    border: 1px solid #fb7185;
    padding: 22px;
    border-radius: 17px;
    color: #fecdd3;
    box-shadow: 0 0 22px #f43f5e22;
}
div[data-testid="stProgress"] > div {
    background-color: #26203e;
}
div[data-testid="stProgress"] div[role="progressbar"] {
    background: linear-gradient(90deg, #8b5cf6, #06b6d4);
}
hr {
    border-color: #37305c;
}
footer, #MainMenu {
    visibility: hidden;
}
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)

    model.load_state_dict(
        torch.load(
            "chest_xray_resnet18.pth",
            map_location=device,
            weights_only=True
        )
    )

    model.eval()
    return model


def make_gradcam(model, image_tensor, original_image):
    activations = []
    gradients = []

    target_layer = model.layer4[-1].conv2

    def forward_hook(module, inputs, output):
        activations.append(output)

    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0])

    h1 = target_layer.register_forward_hook(forward_hook)
    h2 = target_layer.register_full_backward_hook(backward_hook)

    try:
        model.zero_grad()

        output = model(image_tensor)
        predicted_class = output.argmax(dim=1).item()

        output[0, predicted_class].backward()

        activation = activations[0].detach()
        gradient = gradients[0].detach()

        weights = gradient.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu(
            (weights * activation).sum(dim=1)
        ).squeeze()

        cam = cam.cpu().numpy()
        cam = cv2.resize(
            cam,
            (original_image.width, original_image.height)
        )

        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)

        heatmap = np.uint8(255 * cam)
        heatmap = cv2.applyColorMap(
            heatmap,
            cv2.COLORMAP_JET
        )
        heatmap = cv2.cvtColor(
            heatmap,
            cv2.COLOR_BGR2RGB
        )

        original = np.array(original_image.convert("RGB"))

        overlay = cv2.addWeighted(
            original, 0.6, heatmap, 0.4, 0
        )

        return overlay

    finally:
        h1.remove()
        h2.remove()
        model.zero_grad()


transform = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])


st.markdown("""
<div class="hero">
    <h1>🩻 CHEST X-RAY AI</h1>
    <p>✦ DEEP LEARNING • COMPUTER VISION • MEDICAL IMAGING ✦</p>
    <p>Upload an X-ray and explore your trained ResNet18 model's prediction.</p>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="warning">
⚠️ <b>EDUCATIONAL DEMO ONLY</b><br>
This model is not clinically validated. Its predictions and confidence
scores must not be used for diagnosis or healthcare decisions.
</div>
""", unsafe_allow_html=True)


left, right = st.columns([1, 1], gap="large")

with left:
    st.markdown('<div class="panel">', unsafe_allow_html=True)

    st.subheader("📤 Upload X-ray")

    uploaded_file = st.file_uploader(
        "Choose a chest X-ray image",
        type=["jpg", "jpeg", "png"]
    )

    image = None

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")

        st.image(
            image,
            caption="Uploaded chest X-ray",
            use_container_width=True
        )
    else:
        st.info("⬆️ Upload an image to begin analysis.")

    st.markdown('</div>', unsafe_allow_html=True)


with right:
    st.markdown('<div class="panel">', unsafe_allow_html=True)

    st.subheader("🔬 AI Analysis")

    if uploaded_file:
        if st.button("⚡ ANALYSE X-RAY"):
            try:
                model = load_model()
                tensor = transform(image).unsqueeze(0).to(device)

                with st.spinner("Running ResNet18 and generating Grad-CAM..."):
                    outputs = model(tensor)
                    probs = F.softmax(outputs, dim=1)[0].detach()

                    classes = ["NORMAL", "PNEUMONIA"]
                    index = probs.argmax().item()
                    confidence = probs[index].item() * 100

                    heatmap = make_gradcam(
                        model,
                        tensor,
                        image
                    )

                if index == 0:
                    st.markdown(f"""
                    <div class="result-normal">
                        <h2>🟢 NORMAL</h2>
                        <h3>{confidence:.2f}% model confidence</h3>
                        <p>The model classified this image as NORMAL.</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="result-pneumonia">
                        <h2>🔴 PNEUMONIA</h2>
                        <h3>{confidence:.2f}% model confidence</h3>
                        <p>The model classified this image as PNEUMONIA.</p>
                    </div>
                    """, unsafe_allow_html=True)

                st.write("")
                st.markdown("### 📊 Class scores")

                st.caption("NORMAL")
                st.progress(probs[0].item())
                st.write(f"{probs[0].item() * 100:.2f}%")

                st.caption("PNEUMONIA")
                st.progress(probs[1].item())
                st.write(f"{probs[1].item() * 100:.2f}%")

                st.markdown("---")
                st.markdown("### 🔥 Grad-CAM Heatmap")

                st.image(
                    heatmap,
                    caption="Approximate regions influencing the prediction",
                    use_container_width=True
                )

                st.caption(
                    "Grad-CAM is an approximate model-sensitivity visualization, "
                    "not proof of disease or a clinical explanation."
                )

            except Exception as e:
                st.error(f"Analysis failed: {e}")

    else:
        st.info("Your prediction will appear here after you upload an image.")

    st.markdown('</div>', unsafe_allow_html=True)


st.markdown("---")

st.markdown("""
<div style="text-align:center; padding:15px; color:#a78bfa;">
    🟣 Built with PyTorch & Streamlit<br>
    <small>ResNet18 • Transfer Learning • Grad-CAM • Image Classification</small>
</div>
""", unsafe_allow_html=True)
