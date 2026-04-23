import os
import argparse
import cv2
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from modules.feature_extraction import ResNet_FeatureExtractor
from modules.sequence_modeling import BidirectionalLSTM

IMG_W = 224
IMG_H = 64
ALPHABET = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZĐ"

# Preprocessing from train_easyocr_integrated.py
def adjust_contrast_grey(img, target=0.4):
    high = np.percentile(img, 90)
    low  = np.percentile(img, 10)
    contrast = (high-low)/np.maximum(10, high+low)
    if contrast < target:
        img = img.astype(int)
        ratio = 200./np.maximum(10, high-low)
        img = (img - low + 25)*ratio
        img = np.maximum(np.full(img.shape, 0), np.minimum(np.full(img.shape, 255), img)).astype(np.uint8)
    return img

def resize_with_padding(img, target_w=IMG_W, target_h=IMG_H):
    if img is None or img.size == 0:
        return np.zeros((target_h, target_w), dtype=np.uint8)
    h, w = img.shape[:2]
    if h <= 0 or w <= 0:
        return np.zeros((target_h, target_w), dtype=np.uint8)
    scale = min(target_w / w, target_h / h)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    canvas = np.full((target_h, target_w), 255, dtype=np.uint8)
    x0 = (target_w - new_w) // 2
    y0 = (target_h - new_h) // 2
    canvas[y0:y0 + new_h, x0:x0 + new_w] = resized
    return canvas

def preprocess_plate(img):
    if img is None or img.size == 0:
        return np.zeros((IMG_H, IMG_W), dtype=np.uint8)
    img = adjust_contrast_grey(img, target=0.4)
    return resize_with_padding(img, IMG_W, IMG_H)

class LabelConverter:
    def __init__(self, alphabet):
        self.alphabet = alphabet
        self.char_to_dict = {c: i + 1 for i, c in enumerate(alphabet)}
        self.dict_to_char = {i + 1: c for i, c in enumerate(alphabet)}
        self.dict_to_char[0] = "-"
    def decode(self, res):
        if torch.is_tensor(res):
            res = res.tolist()
        out = []
        for i in range(len(res)):
            if res[i] != 0 and (i == 0 or res[i] != res[i - 1]):
                out.append(self.dict_to_char[res[i]])
        return "".join(out)

def vnm_plate_post_process(text):
    import re
    return re.sub(r"[^0-9A-ZĐ]", "", text.upper())

class CRNN(nn.Module):
    def __init__(self, nclass):
        super().__init__()
        self.feature_extraction = ResNet_FeatureExtractor(input_channel=1, output_channel=512)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((None, 1))
        self.sequence_modeling = nn.Sequential(
            BidirectionalLSTM(512, 256, 256),
            BidirectionalLSTM(256, 256, 256)
        )
        self.dropout_fc = nn.Dropout(0.3)
        self.fc = nn.Linear(256, nclass)
    def forward(self, x):
        x = self.feature_extraction(x)
        x = self.adaptive_pool(x.permute(0, 3, 1, 2)).squeeze(3)
        x = self.sequence_modeling(x)
        x = self.dropout_fc(x)
        return self.fc(x)

def decode_ctc_greedy(logits, converter):
    # logits: (N, T, C) or (T, N, C) depending on model
    # Ensure shape is (T, N, C)
    if logits.dim() == 3:
        if logits.shape[0] == 1 and logits.shape[1] > 1:
            logits = logits.squeeze(0)
        # If shape is (N, T, C), permute to (T, N, C)
        if logits.shape[0] < logits.shape[1]:
            logits = logits.permute(1, 0, 2)
    elif logits.dim() == 2:
        logits = logits.unsqueeze(1)  # (T, 1, C)
    probs = torch.softmax(logits, dim=-1)
    max_probs, pred_ids = probs.max(dim=-1)
    # pred_ids: (T, N)
    pred_ids = pred_ids.detach().cpu().numpy()
    max_probs = max_probs.detach().cpu().numpy()
    # Nếu chỉ có 1 ảnh (batch size 1), shape sẽ là (T,)
    if pred_ids.ndim == 1:
        out_chars = []
        out_scores = []
        prev = None
        for t in range(pred_ids.shape[0]):
            pid = pred_ids[t]
            p = max_probs[t]
            if pid != 0 and pid != prev:
                out_chars.append(converter.dict_to_char.get(pid, ""))
                out_scores.append(float(p))
            prev = pid
        text = vnm_plate_post_process("".join(out_chars))
        confidence = float(np.mean(out_scores)) if out_scores else 0.0
        return text, confidence
    else:
        results = []
        for n in range(pred_ids.shape[1]):
            out_chars = []
            out_scores = []
            prev = None
            for t in range(pred_ids.shape[0]):
                pid = pred_ids[t, n]
                p = max_probs[t, n]
                if pid != 0 and pid != prev:
                    out_chars.append(converter.dict_to_char.get(pid, ""))
                    out_scores.append(float(p))
                prev = pid
            text = vnm_plate_post_process("".join(out_chars))
            confidence = float(np.mean(out_scores)) if out_scores else 0.0
            results.append((text, confidence))
        return results[0]

def infer_one(model, converter, image_path, device):
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None, 0.0
    img = preprocess_plate(img)
    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=(0, 1))
    x = torch.from_numpy(img).to(device)
    with torch.no_grad():
        logits = model(x)
        text, conf = decode_ctc_greedy(logits, converter)
    return text, conf

def gather_images(path_obj):
    if path_obj.is_file():
        return [path_obj]
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
    files = [p for p in sorted(path_obj.rglob("*")) if p.is_file() and p.suffix.lower() in exts]
    return files

def parse_args():
    parser = argparse.ArgumentParser(description="Test EasyOCR Integrated Model")
    parser.add_argument(
        "--model",
        type=str,
        default=str(Path(__file__).resolve().parent.parent / "data_ocr" / "best_easyocr_integrated.pt"),
        help="Path to .pt model",
    )
    parser.add_argument("--input", type=str, required=True, help="Path to image or folder containing images")
    parser.add_argument("--cpu", action="store_true", help="Force CPU")
    return parser.parse_args()

def main():
    args = parse_args()
    model_path = Path(args.model)
    input_path = Path(args.input)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    if not input_path.exists():
        raise FileNotFoundError(f"Input path not found: {input_path}")
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    print(f"DEVICE: {device}")
    converter = LabelConverter(ALPHABET)
    model = CRNN(len(ALPHABET) + 1).to(device)
    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    image_files = gather_images(input_path)
    if not image_files:
        raise RuntimeError(f"No images found in: {input_path}")
    print(f"Found {len(image_files)} image(s)")
    for img_path in image_files:
        pred, conf = infer_one(model, converter, img_path, device)
        if pred is None:
            print(f"{img_path} | ERROR: cannot read image")
            continue
        print(f"{img_path} | PRED: {pred} | CONF: {conf:.4f}")

if __name__ == "__main__":
    main()
