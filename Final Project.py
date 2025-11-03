import os
import string
import pandas as pd
import streamlit as st

# ===================================
# Safe Imports
# ===================================
try:
    from PIL import Image
except ImportError:
    st.error("❌ Pillow library issue detected. Please run: pip install pillow==9.5.0")
    st.stop()

try:
    import torch
    import torch.nn as nn
    import torchvision.transforms as T
    from torchvision.models import inception_v3, Inception_V3_Weights
    TORCH_AVAILABLE = True
except Exception as e:
    TORCH_AVAILABLE = False
    st.warning(f"⚠️ PyTorch not fully available. Running in caption-only mode (no model inference). Error: {e}")

import nltk
from nltk.tokenize import RegexpTokenizer

# ===================================
# Utility Functions
# ===================================
def load_descriptions_from_csv(csv_file):
    if not os.path.exists(csv_file):
        st.error(f"❌ Captions file not found: {csv_file}")
        st.stop()
    df = pd.read_csv(csv_file)
    if df.shape[1] < 2:
        st.error("❌ CSV must contain at least 2 columns: image and caption.")
        st.stop()
    image_col, caption_col = df.columns[:2]
    descriptions = {}
    for _, row in df.iterrows():
        image_id = str(row[image_col]).strip()
        caption = str(row[caption_col]).lower().strip()
        descriptions.setdefault(image_id, []).append(f"<start> {caption} <end>")
    return descriptions

def clean_descriptions(descriptions):
    table = str.maketrans('', '', string.punctuation)
    for key, caps in descriptions.items():
        cleaned = []
        for cap in caps:
            words = cap.split()
            words = [w.translate(table) for w in words if w.isalpha()]
            cleaned.append(' '.join(words))
        descriptions[key] = cleaned
    return descriptions

def create_tokenizer(descriptions):
    all_caps = [cap for desc in descriptions.values() for cap in desc]
    tokenizer = RegexpTokenizer(r'\w+')
    vocab = set(word for line in all_caps for word in tokenizer.tokenize(line))
    word2idx = {w: i + 1 for i, w in enumerate(sorted(vocab))}
    idx2word = {i: w for w, i in word2idx.items()}
    return word2idx, idx2word

def max_caption_length(descriptions):
    lengths = [len(cap.split()) for descs in descriptions.values() for cap in descs]
    return max(lengths) if lengths else 0

# ===================================
# Model Definitions
# ===================================
if TORCH_AVAILABLE:
    class EncoderCNN(nn.Module):
        def __init__(self, embed_dim=256):
            super().__init__()
            base = inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1)
            base.aux_logits = False  # Fix for latest torchvision
            base.fc = nn.Identity()
            self.cnn = base
            self.fc = nn.Linear(2048, embed_dim)
        def forward(self, x):
            self.cnn.eval()
            with torch.no_grad():
                features = self.cnn(x)
            return self.fc(features)

    class DecoderRNN(nn.Module):
        def __init__(self, embed_dim, hidden_dim, vocab_size):
            super().__init__()
            self.embed = nn.Embedding(vocab_size, embed_dim)
            self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
            self.fc = nn.Linear(hidden_dim, vocab_size)
        def forward(self, features, captions):
            embeddings = self.embed(captions)
            inputs = torch.cat((features.unsqueeze(1), embeddings[:, :-1]), dim=1)
            outputs, _ = self.lstm(inputs)
            return self.fc(outputs)

# ===================================
# Caption Generation
# ===================================
def generate_caption(decoder, tokenizer, feature, max_len):
    idx2word = {idx: word for word, idx in tokenizer.items()}
    text = ['start']
    for _ in range(max_len):
        seq = [tokenizer.get(w, 0) for w in text]
        seq_tensor = torch.tensor(seq).unsqueeze(0).to(feature.device)
        with torch.no_grad():
            output = decoder(feature, seq_tensor)
        _, predicted = output[:, -1, :].max(1)
        word = idx2word.get(predicted.item(), None)
        if word is None or word == 'end':
            break
        text.append(word)
    return ' '.join(text[1:])

# ===================================
# Streamlit App Configuration
# ===================================
CAPTIONS_FILE = r"D:/GUVI/MINI PROJECT/FINAL PROJECT/captions.txt"
DECODER_PATH = r"D:/GUVI/MINI PROJECT/FINAL PROJECT/decoder.pth"
IMAGES_FOLDER = r"D:/GUVI/MINI PROJECT/FINAL PROJECT/Images"

# ===================================
# Main Streamlit App
# ===================================
def main():
    st.set_page_config(page_title="🖼️ Image Caption Generator", layout="wide")
    st.sidebar.title("🌐 Find Your Way 🌐")
    page = st.sidebar.radio(
        "Go to:",
        ["🏠 Home", "📤 Upload & Generate", "ℹ️ About Project"],
    )

    if page == "🏠 Home":
        st.markdown(
            """
            <h1 style='text-align:center;color:#4A90E2;'>🖼️ Image Caption Generator</h1>
            <p style='text-align:center;font-size:18px;'>
            This application automatically generates captions for images using Deep Learning.<br>
            It combines a CNN encoder and an RNN decoder to understand and describe visual content.
            </p>
            """, unsafe_allow_html=True)
        st.image("https://cdn.pixabay.com/photo/2017/08/07/19/12/ai-2604586_1280.jpg", use_container_width=True)
        st.markdown("---")
        st.info("👈 Use the sidebar to upload an image or learn more about this project.")

    elif page == "📤 Upload & Generate":
        st.markdown("<h2 style='color:#4A90E2;'>📸 Upload an Image to Generate Caption</h2>", unsafe_allow_html=True)
        uploaded = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])
        if uploaded:
            try:
                image = Image.open(uploaded).convert('RGB')
            except Exception:
                st.error("❌ Unable to open the image. Please try another file.")
                st.stop()
            name = os.path.basename(uploaded.name)
            st.image(image, caption=f"Uploaded: {name}", use_container_width=True)
            nltk.download('punkt', quiet=True)
            descriptions = clean_descriptions(load_descriptions_from_csv(CAPTIONS_FILE))
            word2idx, idx2word = create_tokenizer(descriptions)
            max_len = max_caption_length(descriptions)
            vocab_size = len(word2idx) + 1
            st.markdown("---")
            if not TORCH_AVAILABLE:
                if name in descriptions:
                    st.success(f"Dataset Caption: {descriptions[name][0]}")
                else:
                    st.warning("⚠️ PyTorch not available. Displaying dataset caption only.")
                return
            device = torch.device("cpu")
            transform = T.Compose([
                T.Resize((299, 299)),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
            img_tensor = transform(image).unsqueeze(0).to(device)
            encoder = EncoderCNN().to(device)
            feature = encoder(img_tensor)
            if name in descriptions:
                st.success(f"Dataset Caption: {descriptions[name][0]}")
            elif not os.path.exists(DECODER_PATH):
                st.warning("⚠️ Decoder model not found. Please add 'decoder.pth' file.")
            else:
                decoder = DecoderRNN(256, 512, vocab_size).to(device)
                decoder.load_state_dict(torch.load(DECODER_PATH, map_location=device))
                decoder.eval()
                caption = generate_caption(decoder, word2idx, feature, max_len)
                st.success(f"Generated Caption: {caption}")
            st.caption("💡 Tip: Ensure the uploaded image name exactly matches dataset entries.")

    elif page == "ℹ️ About Project":
        st.markdown(
            """
            <h2 style='color:#4A90E2;'>About This Project</h2>
            <p style='font-size:16px;'>
            The <b>Image Caption Generator</b> uses a hybrid Deep Learning architecture:
            </p>
            <ul style='font-size:16px;'>
                <li><b>Encoder:</b> InceptionV3 CNN extracts image features.</li>
                <li><b>Decoder:</b> LSTM RNN generates text descriptions.</li>
                <li><b>Dataset:</b> Flickr8k / Custom Captions Dataset.</li>
                <li><b>Tech Stack:</b> PyTorch, Streamlit, Pandas, NLTK, Pillow.</li>
            </ul>
            <p style='font-size:16px;'>
            Explore, upload images, and see the AI describe what it sees! 🌸
            </p>
            """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
