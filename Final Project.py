import os
import pandas as pd
import string
import torch
import torch.nn as nn
import torchvision.transforms as T
from torchvision.models import inception_v3
from PIL import Image
import streamlit as st
import nltk
from nltk.tokenize import RegexpTokenizer

# =============================
# 1. Load Descriptions from CSV
# =============================
def load_descriptions_from_csv(csv_file):
    if not os.path.exists(csv_file):
        st.error(f"❌ Captions file not found: {csv_file}")
        st.stop()
    df = pd.read_csv(csv_file)
    if df.shape[1] < 2:
        st.error("CSV must contain at least 2 columns: image and caption.")
        st.stop()
    image_col = df.columns[0]
    caption_col = df.columns[1]
    descriptions = {}
    for _, row in df.iterrows():
        image_id = str(row[image_col]).strip()
        caption = str(row[caption_col]).lower().strip()
        descriptions.setdefault(image_id, []).append('<start> ' + caption + ' <end>')
    return descriptions

# =============================
# 2. Text Preprocessing
# =============================
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
    all_captions = [cap for desc in descriptions.values() for cap in desc]
    tokenizer = RegexpTokenizer(r'\w+')
    vocab = set(word for line in all_captions for word in tokenizer.tokenize(line))
    word2idx = {w: i + 1 for i, w in enumerate(sorted(vocab))}
    idx2word = {i: w for w, i in word2idx.items()}
    return word2idx, idx2word

def max_caption_length(descriptions):
    lengths = [len(cap.split()) for descs in descriptions.values() for cap in descs]
    return max(lengths) if lengths else 0

# =============================
# 3. Encoder CNN (Fixed)
# =============================
class EncoderCNN(nn.Module):
    def __init__(self, embed_dim=256):
        super().__init__()
        base = inception_v3(weights='IMAGENET1K_V1', aux_logits=True)
        base.aux_logits = False
        base.AuxLogits = nn.Identity()
        base.fc = nn.Identity()
        self.cnn = base
        self.fc = nn.Linear(2048, embed_dim)

    def forward(self, x):
        self.cnn.eval()
        with torch.no_grad():
            features = self.cnn(x)
            if isinstance(features, tuple):
                features = features[0]
        return self.fc(features)

# =============================
# 4. Decoder RNN
# =============================
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

# =============================
# 5. Caption Generation
# =============================
def generate_caption(decoder, tokenizer, feature, max_len):
    idx2word = {idx: word for word, idx in tokenizer.items()}
    text = ['start']
    for _ in range(max_len):
        sequence = [tokenizer.get(w, 0) for w in text]
        seq_tensor = torch.tensor(sequence).unsqueeze(0).to(feature.device)
        with torch.no_grad():
            output = decoder(feature, seq_tensor)
        _, predicted = output[:, -1, :].max(1)
        word = idx2word.get(predicted.item(), None)
        if word is None or word == 'end':
            break
        text.append(word)
    return ' '.join(text[1:])

# =============================
# 6. Streamlit UI
# =============================
CAPTIONS_FILE = r"D:/GUVI/MINI PROJECT/env/Scripts/Flickr8k_text/captions.csv"
DECODER_PATH = r"D:/GUVI/MINI PROJECT/env/Scripts/Flickr8k_text/decoder.pth"

st.set_page_config(page_title="🧠 Intelligent Caption Generator", layout="centered")
st.markdown("<h1 style='text-align: center; color: #4A90E2;'>🖼️ Image Caption Generator 🖼️</h1>", unsafe_allow_html=True)
st.markdown("---")

uploaded = st.file_uploader("📤 Upload an image", type=["jpg", "jpeg", "png"])

if uploaded:
    image = Image.open(uploaded).convert('RGB')
    image_filename = os.path.basename(uploaded.name).strip()
    st.image(image, caption=f"🖼️ Uploaded: `{image_filename}`", use_container_width=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = T.Compose([
        T.Resize((299, 299)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    img_tensor = transform(image).unsqueeze(0).to(device)

    nltk.download('punkt', quiet=True)

    st.markdown("### 🔄 Loading & Processing...")
    descriptions = load_descriptions_from_csv(CAPTIONS_FILE)
    descriptions = clean_descriptions(descriptions)
    word2idx, idx2word = create_tokenizer(descriptions)
    max_len = max_caption_length(descriptions)
    vocab_size = len(word2idx) + 1

    encoder = EncoderCNN().to(device)
    feature = encoder(img_tensor)

    st.markdown("---")
    if image_filename in descriptions:
        matched_caption = descriptions[image_filename][0]
        st.markdown("### ✅ Caption found in dataset:")
        st.success(f"**📝 {matched_caption}**")
    elif not os.path.exists(DECODER_PATH):
        st.warning("⚠️ Trained decoder model not found. Please add 'decoder.pth' in the path.")
    else:
        decoder = DecoderRNN(embed_dim=256, hidden_dim=512, vocab_size=vocab_size).to(device)
        decoder.load_state_dict(torch.load(DECODER_PATH, map_location=device))
        decoder.eval()
        caption = generate_caption(decoder, word2idx, feature, max_len)
        st.markdown("### ✨ Generated Caption:")
        st.success(f"**📝 {caption}**")

    st.markdown("---")
    st.caption("💡 Tip: For best results, ensure image name matches exactly with the dataset.")
