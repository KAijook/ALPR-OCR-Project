"""
Vietnamese License Plate OCR Training - EasyOCR Integrated

Architecture: ResNet backbone + BiLSTM + CTC Loss
Preprocessing: EasyOCR contrast adjustment
Features:
- Full EasyOCR integration
- Automatic contrast adjustment based on percentiles
- Contrast adjustment during batch collation
- NormalizePAD for better image handling
- Most advanced preprocessing

Expected performance: Best preprocessing, highest accuracy
"""

import os
import cv2
import random
import re
import hashlib
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import GroupShuffleSplit
import editdistance
import math
from PIL import Image
import torchvision.transforms as transforms

from modules.feature_extraction import ResNet_FeatureExtractor
from modules.sequence_modeling import BidirectionalLSTM


ROOT = r"C:\Users\PC\visual_code\.cph\.vscode\content\data_ocr"
IMG_W = 224
IMG_H = 64
ALPHABET = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZĐ"


# EasyOCR preprocessing functions
def contrast_grey(img):
    high = np.percentile(img, 90)
    low  = np.percentile(img, 10)
    return (high-low)/np.maximum(10, high+low), high, low

def adjust_contrast_grey(img, target=0.4):
    contrast, high, low = contrast_grey(img)
    if contrast < target:
        img = img.astype(int)
        ratio = 200./np.maximum(10, high-low)
        img = (img - low + 25)*ratio
        img = np.maximum(np.full(img.shape, 0) ,np.minimum(np.full(img.shape, 255), img)).astype(np.uint8)
    return img


class NormalizePAD(object):
    def __init__(self, max_size, PAD_type='right'):
        self.toTensor = transforms.ToTensor()
        self.max_size = max_size
        self.max_width_half = math.floor(max_size[2] / 2)
        self.PAD_type = PAD_type

    def __call__(self, img):
        img = self.toTensor(img)
        img.sub_(0.5).div_(0.5)
        c, h, w = img.size()
        Pad_img = torch.FloatTensor(*self.max_size).fill_(0)
        Pad_img[:, :, :w] = img  # right pad
        if self.max_size[2] != w:  # add border Pad
            Pad_img[:, :, w:] = img[:, :, w - 1].unsqueeze(2).expand(c, h, self.max_size[2] - w)
        return Pad_img


class AlignCollate(object):
    def __init__(self, imgH=32, imgW=100, keep_ratio_with_pad=False, adjust_contrast=0.):
        self.imgH = imgH
        self.imgW = imgW
        self.keep_ratio_with_pad = keep_ratio_with_pad
        self.adjust_contrast = adjust_contrast

    def __call__(self, batch):
        batch = filter(lambda x: x is not None, batch)
        images, labels = zip(*batch)

        resized_max_w = self.imgW
        input_channel = 1
        transform = NormalizePAD((input_channel, self.imgH, resized_max_w))

        resized_images = []
        for image in images:
            w, h = image.size
            #### augmentation here - change contrast
            if self.adjust_contrast > 0:
                image = np.array(image.convert("L"))
                image = adjust_contrast_grey(image, target=self.adjust_contrast)
                image = Image.fromarray(image, 'L')

            ratio = w / float(h)
            if math.ceil(self.imgH * ratio) > self.imgW:
                resized_w = self.imgW
            else:
                resized_w = math.ceil(self.imgH * ratio)

            resized_image = image.resize((resized_w, self.imgH), Image.BICUBIC)
            resized_images.append(transform(resized_image))

        image_tensors = torch.cat([t.unsqueeze(0) for t in resized_images], 0)
        return image_tensors


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


def direct_resize(img, target_w=IMG_W, target_h=IMG_H):
    if img is None or img.size == 0:
        return np.zeros((target_h, target_w), dtype=np.uint8)
    return cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_CUBIC)


def estimate_skew_angle(img):
    if img is None or img.size == 0:
        return 0.0

    blur = cv2.GaussianBlur(img, (3, 3), 0)
    edges = cv2.Canny(blur, 60, 180)
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=45,
        minLineLength=max(20, int(min(img.shape[:2]) * 0.35)),
        maxLineGap=10,
    )
    if lines is None:
        return 0.0

    angles = []
    weights = []
    for line in lines[:, 0, :]:
        x1, y1, x2, y2 = line
        dx = float(x2 - x1)
        dy = float(y2 - y1)
        length = float(np.hypot(dx, dy))
        if length < 8:
            continue

        angle = float(np.degrees(np.arctan2(dy, dx)))
        if angle < -90:
            angle += 180
        elif angle > 90:
            angle -= 180
        if abs(angle) > 40:
            continue

        angles.append(angle)
        weights.append(length)

    if not angles:
        return 0.0

    return float(np.average(np.array(angles), weights=np.array(weights)))


def deskew_plate(img, max_abs_angle=25.0):
    if img is None or img.size == 0:
        return np.zeros((IMG_H, IMG_W), dtype=np.uint8)

    angle = estimate_skew_angle(img)
    if abs(angle) < 0.5 or abs(angle) > max_abs_angle:
        return img

    h, w = img.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, 1.0)
    return cv2.warpAffine(
        img,
        m,
        (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=255,
    )


def enhance_small_plate(img):
    if img is None or img.size == 0:
        return np.zeros((IMG_H, IMG_W), dtype=np.uint8)

    h, w = img.shape[:2]
    min_side = min(h, w)
    area = h * w
    if min_side >= 56 and area >= 5000:
        return img

    target_min_side = 80.0 if min_side < 32 else 64.0
    scale = float(np.clip(target_min_side / max(1.0, float(min_side)), 1.0, 4.0))

    if scale > 1.01:
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))
        up = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
    else:
        up = img

    up = cv2.bilateralFilter(up, d=5, sigmaColor=20, sigmaSpace=20)
    blur = cv2.GaussianBlur(up, (0, 0), 0.8)
    up = cv2.addWeighted(up, 1.12, blur, -0.12, 0)
    return up


def preprocess_tiny_plate(img):
    if img is None or img.size == 0:
        return np.zeros((IMG_H, IMG_W), dtype=np.uint8)

    h, w = img.shape[:2]
    min_side = min(h, w)
    area = h * w

    if min_side >= 40 and area >= 2500:
        return resize_with_padding(img, IMG_W, IMG_H)

    img = enhance_small_plate(img)
    if abs(estimate_skew_angle(img)) >= 6.0:
        img = deskew_plate(img)

    return resize_with_padding(img, IMG_W, IMG_H)


def preprocess_plate(img):
    if img is None or img.size == 0:
        return np.zeros((IMG_H, IMG_W), dtype=np.uint8)

    # Use EasyOCR contrast adjustment
    img = adjust_contrast_grey(img, target=0.4)

    # Keep deskew if needed
    if abs(estimate_skew_angle(img)) >= 6.0:
        img = deskew_plate(img)

    return resize_with_padding(img, IMG_W, IMG_H)


def augment_plate(img):
    if random.random() < 0.2:
        img = cv2.GaussianBlur(img, (3, 3), 0)

    if random.random() < 0.25:
        alpha = random.uniform(0.9, 1.12)
        beta = random.uniform(-12, 12)
        img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

    if random.random() < 0.2:
        h, w = img.shape[:2]
        src = np.float32([[0, 0], [w - 1, 0], [0, h - 1], [w - 1, h - 1]])
        jitter = min(w, h) * 0.04
        dst = src + np.random.uniform(-jitter, jitter, src.shape).astype(np.float32)
        m = cv2.getPerspectiveTransform(src, dst)
        img = cv2.warpPerspective(img, m, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

    if random.random() < 0.15:
        noise = np.random.normal(0, 5, img.shape).astype(np.float32)
        img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    return img


def vnm_plate_post_process(text):
    return re.sub(r"[^0-9A-ZĐ]", "", text.upper())


CONFUSABLE_CHARS = set("OQDILZSGB01258")
HARD_TOKENS = {
    "HC", "TM", "FH", "KB", "QH", "KV", "TK", "CK", "VK", "QA", "QK", "VT", "TN", "TH"
}


def sample_weight_for_label(label):
    label = vnm_plate_post_process(label)
    if len(label) == 0:
        return 1.0

    confusable_count = sum(1 for ch in label if ch in CONFUSABLE_CHARS)
    confusable_ratio = confusable_count / len(label)
    hard_token_boost = 0.6 if any(tok in label for tok in HARD_TOKENS) else 0.0

    length_boost = 0.0
    if len(label) >= 10:
        length_boost = 1.0
    elif len(label) == 9:
        length_boost = 0.75
    elif len(label) == 8:
        length_boost = 0.45
    elif len(label) == 7:
        length_boost = 0.20

    prefix_boost = 0.25 if len(label) >= 2 and label[0].isalpha() and label[1].isalpha() else 0.0

    return float(min(3.4, 1.0 + confusable_ratio * 1.2 + hard_token_boost + length_boost + prefix_boost))


class LabelConverter:
    def __init__(self, alphabet):
        self.alphabet = alphabet
        self.char_to_dict = {c: i + 1 for i, c in enumerate(alphabet)}
        self.dict_to_char = {i + 1: c for i, c in enumerate(alphabet)}
        self.dict_to_char[0] = "-"

    def encode(self, text):
        return [self.char_to_dict[c] for c in text if c in self.char_to_dict]

    def decode(self, res):
        if torch.is_tensor(res):
            res = res.tolist()
        out = []
        for i in range(len(res)):
            if res[i] != 0 and (i == 0 or res[i] != res[i - 1]):
                out.append(self.dict_to_char[res[i]])
        return "".join(out)


class OCRDataset(Dataset):
    def __init__(self, df, converter, train=True):
        self.samples = []
        self.converter = converter
        self.train = train

        for _, row in df.iterrows():
            self.samples.append((row["Path"], row["Label"]))

        print(f"[DATASET] {'train' if train else 'val'} samples: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        name = os.path.basename(path)

        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            img = np.zeros((IMG_H, IMG_W), dtype=np.uint8)

        img = preprocess_plate(img)
        if self.train:
            img = augment_plate(img)

        img = img.astype(np.float32) / 255.0
        img = np.expand_dims(img, 0)

        target = self.converter.encode(label)
        sample_weight = sample_weight_for_label(label)

        return torch.from_numpy(img), torch.tensor(target), len(target), sample_weight, name


def load_bike_extra_df():
    label_dir = os.path.join(ROOT, "biensoxemayhon100bien", "label")
    image_dir = os.path.join(ROOT, "biensoxemayhon100bien", "cropped")

    if not os.path.isdir(label_dir) or not os.path.isdir(image_dir):
        return pd.DataFrame(columns=["Name", "Label", "Type"])

    rows = []
    for fname in os.listdir(label_dir):
        if not fname.lower().endswith(".txt"):
            continue

        txt_path = os.path.join(label_dir, fname)
        try:
            with open(txt_path, "r", encoding="utf-8") as f:
                label = f.readline().strip().replace(" ", "").upper()
        except OSError:
            continue

        if len(label) == 0:
            continue

        stem, _ = os.path.splitext(fname)
        img_name = f"{stem}.jpg"
        if not os.path.exists(os.path.join(image_dir, img_name)):
            continue

        rows.append({"Name": img_name, "Label": label, "Type": "bike_extra"})

    print(f"[DATASET] bike_extra pairs loaded: {len(rows)}")
    return pd.DataFrame(rows, columns=["Name", "Label", "Type"])


def build_usable_df(df, min_keep_side=24, min_keep_area=800):
    folder_map = {
        "crop": "cropped",
        "gen": "generated",
        "bike_extra": os.path.join("biensoxemayhon100bien", "cropped"),
    }

    rows = []
    seen_groups = set()
    dup_groups = 0

    for _, row in df.iterrows():
        folder = folder_map.get(str(row["Type"]).lower(), "")
        path = os.path.join(ROOT, folder, str(row["Name"]).strip())

        if not os.path.exists(path):
            continue

        label = str(row["Label"]).strip().replace(" ", "").upper()
        if len(label) == 0:
            continue

        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None or img.size == 0:
            continue

        group_hash = hashlib.md5(img.tobytes()).hexdigest()
        if group_hash in seen_groups:
            dup_groups += 1
        else:
            seen_groups.add(group_hash)

        rows.append({
            "Name": row["Name"],
            "Label": label,
            "Type": row["Type"],
            "Path": path,
            "GroupHash": group_hash,
        })

    print(f"[DATASET] duplicate image groups seen: {dup_groups}")
    print(f"[DATASET] remaining usable rows: {len(rows)}")
    return pd.DataFrame(rows)


def collate_fn(batch):
    imgs, targets, lengths, weights, names = zip(*batch)

    # Apply EasyOCR contrast adjustment during training
    processed_imgs = []
    for img in imgs:
        img_np = img.squeeze(0).numpy() * 255  # Convert back to numpy
        img_np = img_np.astype(np.uint8)
        img_np = adjust_contrast_grey(img_np, target=0.4)
        img_tensor = torch.from_numpy(img_np.astype(np.float32) / 255.0).unsqueeze(0)
        processed_imgs.append(img_tensor)

    imgs = torch.stack(processed_imgs)
    targets = torch.cat(targets)
    lengths = torch.tensor(lengths)
    weights = torch.tensor(weights, dtype=torch.float32)
    return imgs, targets, lengths, weights, list(names)


def decode_target_batch(targets, lengths, converter):
    texts = []
    start = 0
    for ln in lengths.tolist():
        seq = targets[start:start + ln].tolist()
        texts.append("".join(converter.dict_to_char[i] for i in seq if i != 0))
        start += ln
    return texts


def cer(pred, gt):
    if len(gt) == 0:
        return 0.0 if len(pred) == 0 else 1.0
    return editdistance.eval(pred, gt) / len(gt)


class CRNN(nn.Module):
    def __init__(self, nclass):
        super().__init__()

        # Use ResNet backbone from EasyOCR instead of simple CNN
        self.feature_extraction = ResNet_FeatureExtractor(input_channel=1, output_channel=512)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((None, 1))

        # Use BidirectionalLSTM from EasyOCR instead of basic LSTM
        self.sequence_modeling = nn.Sequential(
            BidirectionalLSTM(512, 256, 256),
            BidirectionalLSTM(256, 256, 256)
        )

        self.dropout_fc = nn.Dropout(0.3)
        self.fc = nn.Linear(256, nclass)

    def forward(self, x):
        # Feature extraction with ResNet
        x = self.feature_extraction(x)
        # Permute and pool like in EasyOCR
        x = self.adaptive_pool(x.permute(0, 3, 1, 2)).squeeze(3)

        # Sequence modeling with BiLSTM
        x = self.sequence_modeling(x)

        x = self.dropout_fc(x)
        return self.fc(x)


def get_loaders(converter, batch_size=16, min_keep_side=24, min_keep_area=800):
    df1 = pd.read_csv(os.path.join(ROOT, "labels/gen_labels.csv"))
    df2 = pd.read_csv(os.path.join(ROOT, "labels/crop_labels.csv"))

    df1["Type"] = "gen"
    df2["Type"] = "crop"
    df = pd.concat([df1, df2], ignore_index=True)

    bike_extra_df = load_bike_extra_df()
    if len(bike_extra_df) > 0:
        df = pd.concat([df, bike_extra_df], ignore_index=True)

    df = build_usable_df(df, min_keep_side=min_keep_side, min_keep_area=min_keep_area)
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, val_idx = next(splitter.split(df, groups=df["GroupHash"]))

    train_df = df.iloc[train_idx].reset_index(drop=True)
    val_df = df.iloc[val_idx].reset_index(drop=True)

    train_ds = OCRDataset(train_df, converter, train=True)
    val_ds = OCRDataset(val_df, converter, train=False)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=True,
    )

    return train_loader, val_loader


def train(epochs=100, batch_size=16, min_keep_side=24, min_keep_area=800, early_stop_patience=8):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("DEVICE:", device)

    converter = LabelConverter(ALPHABET)
    train_loader, val_loader = get_loaders(
        converter,
        batch_size=batch_size,
        min_keep_side=min_keep_side,
        min_keep_area=min_keep_area,
    )

    model = CRNN(len(ALPHABET) + 1).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=5e-5)
    criterion = nn.CTCLoss(blank=0, zero_infinity=True, reduction="none")
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=3,
        threshold=1e-3,
        min_lr=1e-6,
    )

    save_path = os.path.join(ROOT, "best_easyocr_integrated.pt")
    best_acc = -1.0
    no_improve = 0
    early_stop_patience = int(early_stop_patience)

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        for i, (imgs, targets, lengths, weights, _) in enumerate(train_loader):
            imgs = imgs.to(device)
            targets = targets.to(device)
            weights = weights.to(device)

            preds = model(imgs)
            preds_ctc = preds.permute(1, 0, 2).contiguous()
            input_lengths = torch.full(
                size=(imgs.size(0),),
                fill_value=preds_ctc.size(0),
                dtype=torch.long,
                device=device,
            )

            per_sample_loss = criterion(preds_ctc.log_softmax(2), targets, input_lengths, lengths)
            loss = (per_sample_loss * weights).sum() / weights.sum().clamp_min(1e-6)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            total_loss += loss.item()
            if i % 50 == 0:
                print(f"Batch {i} Loss {loss.item():.4f}")

        model.eval()
        correct = 0
        total = 0
        char_correct = 0
        char_total = 0

        with torch.no_grad():
            for imgs, targets, lengths, _, _ in val_loader:
                imgs = imgs.to(device)
                preds = model(imgs)
                pred_ids = preds.argmax(2)

                pred_texts = [vnm_plate_post_process(converter.decode(seq)) for seq in pred_ids]
                gt_texts = [vnm_plate_post_process(t) for t in decode_target_batch(targets, lengths, converter)]

                for pred_t, gt_t in zip(pred_texts, gt_texts):
                    if pred_t == gt_t:
                        correct += 1
                    total += 1

                    for p_c, g_c in zip(pred_t, gt_t):
                        char_total += 1
                        if p_c == g_c:
                            char_correct += 1

                    if len(gt_t) > len(pred_t):
                        char_total += len(gt_t) - len(pred_t)
                    elif len(pred_t) > len(gt_t):
                        char_total += len(pred_t) - len(gt_t)

        val_acc = (correct / total) if total > 0 else 0.0
        char_acc = (char_correct / char_total) if char_total > 0 else 0.0
        scheduler.step(val_acc)
        current_lr = optimizer.param_groups[0]["lr"]

        if val_acc > best_acc:
            best_acc = val_acc
            no_improve = 0
            torch.save(model.state_dict(), save_path)
            print(f"[BEST] Saved model with Val Acc {val_acc * 100:.2f}% -> {save_path}")
        else:
            no_improve += 1

        print(
            f"Epoch {epoch + 1} DONE | Loss {total_loss:.4f} | "
            f"Val Acc {val_acc * 100:.2f}% | Char Acc {char_acc * 100:.2f}% | "
            f"LR {current_lr:.6f} | NoImprove {no_improve}/{early_stop_patience}"
        )

        if no_improve >= early_stop_patience:
            print(f"[EARLY STOP] No Val Acc improvement for {early_stop_patience} epochs.")
            break


def parse_args():
    parser = argparse.ArgumentParser(description="Train OCR from scratch and filter too-small images")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--early-stop-patience", type=int, default=8)
    parser.add_argument("--min-keep-side", type=int, default=24)
    parser.add_argument("--min-keep-area", type=int, default=800)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(
        epochs=args.epochs,
        batch_size=args.batch_size,
        early_stop_patience=args.early_stop_patience,
        min_keep_side=args.min_keep_side,
        min_keep_area=args.min_keep_area,
    )
