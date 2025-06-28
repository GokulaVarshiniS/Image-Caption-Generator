🖼️ Image Caption Generator 🖼️
📘 README: Image Caption Generator
Overview: This Streamlit app allows users to upload an image and receive an AI-generated caption. It uses a pre-trained InceptionV3 CNN encoder and an RNN decoder trained on Flickr8k-style captions.

🧠 How It Works:
Upload an Image: Drag and drop or browse a .jpg, .jpeg, or .png file.

Match Image to Captions (CSV): If your uploaded image filename matches an entry in captions.csv, that caption will be displayed immediately.

Fallback to Generation: If no match is found and a trained decoder is available (decoder.pth), it generates a caption using the encoder-decoder model.

⚙️ Requirements:
Python 3.7+

Streamlit

PyTorch

Torchvision

pandas, PIL, nltk

🛠️ Files Needed:
captions.csv: Must include two columns — image filename and caption.

decoder.pth: Trained model weights for the RNN decoder.

A trained image captioning model with consistent vocabulary.

🐛 Troubleshooting:
ValueError about batch norm: Make sure the encoder sets aux_logits = False or replaces AuxLogits with nn.Identity().

FileNotFoundError: Double-check the paths to captions.csv and decoder.pth.
