# ◈ Truth vs. Illusion

**AI image authenticity detection.** Upload an image and the app estimates whether it is AI-generated or real, using a fine-tuned EfficientNet-B0 classifier. When the model isn't confident, it says so instead of guessing.

Built with PyTorch, torchvision and Streamlit.

---

## Features

- **Image detection**: upload a JPG, JPEG, PNG or WEBP image and get a verdict with a confidence score.
- **Uncertainty handling**: predictions below 75% confidence are reported as `UNCERTAIN` rather than forced into FAKE or REAL.
- **Chat-style interface**: dark-themed Streamlit UI with drag-and-drop uploads and a "New Analysis" reset.

> **Status:** Only image detection is implemented. Text detection appears in the UI but is not yet wired up to a model.

---

## How it works

1. **Training** (`train_image.py`): fine-tunes a pretrained EfficientNet-B0 on the [CIFAKE](https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images) dataset. The feature extractor is frozen and only the final classification layer is trained.
2. **Inference** (`detector.py`): loads the saved checkpoint, preprocesses the image, and returns a label plus softmax confidence.
3. **App** (`app.py`): Streamlit front end that passes uploaded images to `predict_image()` and displays the result.

### Decision logic

| Model output | Confidence | Result shown |
|---|---|---|
| FAKE | ≥ 75% | YES — likely AI-generated |
| REAL | ≥ 75% | NO — likely not AI-generated |
| Either | < 75% | UNCERTAIN — insufficient evidence |

---

## Project structure

```
.
├── app.py             # Streamlit web app
├── detector.py        # Model loading and prediction
├── train_image.py     # Training script
├── requirements.txt   # Python dependencies
├── data/
│   └── cifake/
│       ├── train/
│       │   ├── FAKE/
│       │   └── REAL/
│       └── test/
│           ├── FAKE/
│           └── REAL/
└── models/
    ├── image_detector.pth    # Created by training
    └── image_classes.json    # Created by training
```

---

## Getting started

### 1. Clone and install

```bash
git clone <your-repo-url>
cd <your-repo-folder>

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

Suggested `requirements.txt`:

```
torch
torchvision
streamlit>=1.43
pillow
```

Streamlit 1.43 or newer is needed for `st.chat_input(accept_file=True)`.

### 2. Download the dataset

Download CIFAKE from Kaggle and extract it so the folders match the structure above (`data/cifake/train/{FAKE,REAL}` and `data/cifake/test/{FAKE,REAL}`).

### 3. Train the model

```bash
python train_image.py
```

This saves `models/image_detector.pth` and `models/image_classes.json`.

Training settings are at the top of `train_image.py`:

| Setting | Default | Notes |
|---|---|---|
| `BATCH_SIZE` | 32 | |
| `EPOCHS` | 5 | |
| `LEARNING_RATE` | 0.0001 | |
| `MAX_TRAIN_PER_CLASS` | 1000 | Subset for a fast first run; raise it for better accuracy |
| `MAX_TEST_PER_CLASS` | 200 | |

A GPU is used automatically if available; otherwise training runs on CPU.

### 4. Run the app

```bash
streamlit run app.py
```

Then open the URL Streamlit prints (usually http://localhost:8501) and upload an image.

---

## Limitations

- **Dataset bias**: CIFAKE consists of small, low-resolution images (originally 32×32). The model may not generalize well to high-resolution photos or to images from newer generators.
- **Not proof**: results are probabilistic. A confident prediction is evidence, not certainty, and should not be used as the sole basis for important decisions.
- **Small default training run**: the default settings use a subset of the data and a frozen backbone, so accuracy is modest until you train on more data or unfreeze layers.
- **Images only**: no text, video or audio detection yet.

---

## Roadmap

- [ ] Train on the full CIFAKE dataset
- [ ] Fine-tune the EfficientNet backbone (unfreeze later layers)
- [ ] Evaluate on higher-resolution, more diverse images
- [ ] Add text detection (AI-generated vs. human-written)
- [ ] Persist analysis history

---

## Acknowledgements

- [CIFAKE dataset](https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images) by Bird & Lotfi
- [EfficientNet](https://arxiv.org/abs/1905.11946) via torchvision
- [Streamlit](https://streamlit.io/)

.
