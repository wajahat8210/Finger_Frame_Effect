import cv2
import numpy as np
import random

def effect_pixelate(img, block=12):
    h, w = img.shape[:2]
    if h < 1 or w < 1:
        return img
    small = cv2.resize(img, (max(1, w // block), max(1, h // block)), interpolation=cv2.INTER_LINEAR)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
def effect_blur(img):
    return cv2.GaussianBlur(img, (25, 25), 0)
def effect_invert(img):
    return cv2.bitwise_not(img)
def effect_noir(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.convertScaleAbs(gray, alpha=1.3, beta=-20)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
def effect_glitch(img):
    b, g, r = cv2.split(img)
    shift = 8
    b = np.roll(b, shift, axis=1)
    r = np.roll(r, -shift, axis=1)
    out = cv2.merge([b, g, r])
    h = out.shape[0]
    for _ in range(4):
        y = random.randint(0, max(0, h - 10))
        slice_h = random.randint(4, 10)
        dx = random.randint(-15, 15)
        out[y:y + slice_h] = np.roll(out[y:y + slice_h], dx, axis=1)
    return out
def effect_toon(img):
    color = cv2.bilateralFilter(img, d=9, sigmaColor=75, sigmaSpace=75)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_blur = cv2.medianBlur(gray, 5)
    edges = cv2.adaptiveThreshold(gray_blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 9)
    edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    return cv2.bitwise_and(color, edges_colored)
def effect_van_gogh(img):
    painted = cv2.stylization(img, sigma_s=60, sigma_r=0.45)
    hsv = cv2.cvtColor(painted, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[..., 1] = np.clip(hsv[..., 1] * 1.4, 0, 255)
    hsv = hsv.astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
EFFECTS = {
    ord('1'): ("Pixelate", effect_pixelate),
    ord('2'): ("Blur", effect_blur),
    ord('3'): ("Invert", effect_invert),
    ord('4'): ("Noir", effect_noir),
    ord('5'): ("Glitch", effect_glitch),
    ord('6'): ("Toon", effect_toon),
    ord('7'): ("Van Gogh", effect_van_gogh),
}
def apply_effect_in_quad(frame, quad_int, effect_fn):
    """Apply effect_fn only inside the quad region; leave the rest of the frame untouched."""
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    cv2.fillConvexPoly(mask, quad_int, 255)
    x, y, w, h = cv2.boundingRect(quad_int)
    x, y = max(x, 0), max(y, 0)
    w = min(w, frame.shape[1] - x)
    h = min(h, frame.shape[0] - y)
    if w <= 0 or h <= 0:
        return frame
    crop = frame[y:y + h, x:x + w]
    effect_crop = effect_fn(crop)
    mask_crop = mask[y:y + h, x:x + w]
    region = frame[y:y + h, x:x + w]
    region[mask_crop == 255] = effect_crop[mask_crop == 255]
    frame[y:y + h, x:x + w] = region
    return frame