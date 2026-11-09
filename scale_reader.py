import cv2
import numpy as np
import os
import re
import json
import time

# ---------- تنظیمات ----------
IP = "192.168.3.36"
USER = "admin"
PASSWORD = os.environ.get("CAM_PASS", "")  # رمز از متغیر محیطی خوانده می‌شود
CHANNEL = 1
SUBTYPE = 0  # 0 = کیفیت بالا، 1 = کیفیت پایین
URL = f"rtsp://{USER}:{PASSWORD}@{IP}:554/cam/realmonitor?channel={CHANNEL}&subtype={SUBTYPE}"

ROI_FILE = "roi.json"  # ناحیه نمایشگر ذخیره می‌شود
SHOW_WINDOW = True

# ---------- ذخیره و خواندن ناحیه نمایشگر ----------
def load_roi():
    if os.path.exists(ROI_FILE):
        with open(ROI_FILE, "r", encoding="utf-8") as f:
            return tuple(json.load(f))
    return None

def save_roi(roi):
    with open(ROI_FILE, "w", encoding="utf-8") as f:
        json.dump(list(roi), f)

# ---------- تشخیص ستون‌های روشن (قرمز) ----------
def red_mask(roi_img):
    hsv = cv2.cvtColor(roi_img, cv2.COLOR_BGR2HSV)
    m1 = cv2.inRange(hsv, (0, 100, 140), (10, 255, 255))
    m2 = cv2.inRange(hsv, (170, 100, 140), (180, 255, 255))
    return m1 | m2

def group_columns(mask, gap=3):
    cols = np.where(mask.sum(axis=0) > 0)[0]
    groups = []
    if len(cols) == 0:
        return groups
    start = prev = cols[0]
    for c in cols[1:]:
        if c - prev > gap:
            groups.append((start, prev))
            start = c
        prev = c
    groups.append((start, prev))
    return groups

def read_digits(roi_img):
    """
    خواندن ساده: هر گروه روشن را با نسبت ارتفاع/عرض و تعداد پیکسل
    به‌صورت تقریبی تشخیص می‌دهد. نقطه‌ها کوچک‌ترند و از همین راه جدا می‌شوند.
    برای دقت بیشتر، بعد از جمع‌آوری نمونه‌ها می‌توان جدول الگو ساخت.
    """
    mask = red_mask(roi_img)
    groups = group_columns(mask)
    chars = []
    for (x1, x2) in groups:
        sub = mask[:, x1:x2 + 1]
        rows = np.where(sub.sum(axis=1) > 0)[0]
        if len(rows) == 0:
            continue
        h = rows[-1] - rows[0] + 1
        w = x2 - x1 + 1
        area = int(sub.sum() // 255)
        chars.append({"x": x1, "y": rows[0], "h": h, "w": w, "area": area})
    return chars, mask

# ---------- انتخاب ناحیه با ماوس ----------
def select_roi(cap):
    ret, frame = cap.read()
    if not ret:
        raise RuntimeError("دریافت تصویر از دوربین ممکن نیست.")
    print("ناحیه نمایشگر را با ماوس انتخاب کنید و Enter بزنید.")
    roi = cv2.selectROI("Select display", frame, False)
    cv2.destroyWindow("Select display")
    save_roi(roi)
    return roi

# ---------- برنامه اصلی ----------
def main():
    if not PASSWORD:
        print("متغیر CAM_PASS تنظیم نشده است. ابتدا مرحله ۲ را انجام دهید.")
        return

    cap = cv2.VideoCapture(URL)
    if not cap.isOpened():
        print("اتصال به دوربین برقرار نشد. IP، رمز، RTSP و شبکه را بررسی کنید.")
        return

    roi = load_roi()
    if roi is None:
        roi = select_roi(cap)

    x, y, w, h = roi
    last_info = None

    while True:
        ret, frame = cap.read()
        if not ret:
            print("خطا در دریافت تصویر، تلاش مجدد...")
            time.sleep(1)
            cap.release()
            cap = cv2.VideoCapture(URL)
            continue

        crop = frame[y:y + h, x:x + w]
        chars, mask = read_digits(crop)

        info = f"تعداد بخش‌های روشن: {len(chars)}"
        if info != last_info:
            print(info)
            last_info = info

        if SHOW_WINDOW:
            display = crop.copy()
            cv2.putText(display, info, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.imshow("Scale", display)
            cv2.imshow("Mask", mask)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        if key == ord('r'):  # دوباره انتخاب ناحیه
            roi = select_roi(cap)
            x, y, w, h = roi

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
