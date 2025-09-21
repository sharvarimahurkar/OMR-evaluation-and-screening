# evaluate.py
import cv2
import numpy as np
import csv
import os

# ========== CONFIG ==========
image_path = "omr_cropped.jpg"   # output of warp.py
# Fill the actual answer key (1..100). Example below for first 10:
answer_key = {i: None for i in range(1,101)}
# Example answers (replace with real ones)
answer_key.update({
    1:"B",2:"C",3:"A",4:"D",5:"A",6:"B",7:"C",8:"A",9:"D",10:"B"
    # continue up to 100: answer_key[11]="A", ...
})

OPTIONS = 4   # A-D
SUBJECTS = 5  # 5 columns (20 questions each)
Q_PER_SUB = 20

# ========== LOAD & PREPROCESS ==========
if not os.path.exists(image_path):
    raise FileNotFoundError(f"Run warp.py first to create {image_path}")

img = cv2.imread(image_path)
if img is None:
    raise FileNotFoundError("Cannot open warped image")

# Preprocess for consistent detection
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (5,5), 0)
thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

# ========== CALIBRATION ==========
print("Calibration: click 4 bubbles in this order:")
print("1) Q1 - Option A   2) Q1 - Option B   3) Q2 - Option A   4) Q21 - Option A")

calib = []
clone = img.copy()
def on_click(e, x, y, flags, param):
    if e == cv2.EVENT_LBUTTONDOWN:
        calib.append((x,y))
        cv2.circle(clone, (x,y), 6, (255,0,0), -1)
        cv2.imshow("Calibration", clone)

cv2.imshow("Calibration", clone)
cv2.setMouseCallback("Calibration", on_click)
cv2.waitKey(0)
cv2.destroyAllWindows()

if len(calib) != 4:
    raise Exception("Calibration requires exactly 4 clicks. Rerun and click 4 bubbles.")

Q1A = calib[0]
Q1B = calib[1]
Q2A = calib[2]
Q21A = calib[3]

x_gap = abs(Q1B[0] - Q1A[0])
y_gap = abs(Q2A[1] - Q1A[1])
col_offset = abs(Q21A[0] - Q1A[0])

print(f"Calibration values: x_gap={x_gap}, y_gap={y_gap}, col_offset={col_offset}")

# ========== DETECTION ==========
answers = {}
debug = img.copy()
r = max(8, int(min(x_gap,y_gap)//2 - 1))  # ROI half-size (safety)
if r < 4:
    r = 8

for q in range(1, SUBJECTS*Q_PER_SUB + 1):
    section = (q-1) // Q_PER_SUB   # 0..4
    idx = (q-1) % Q_PER_SUB       # 0..19

    base_x = Q1A[0] + section * col_offset
    base_y = Q1A[1] + idx * y_gap

    centers = []
    fills = []
    for opt in range(OPTIONS):
        cx = int(base_x + opt * x_gap)
        cy = int(base_y)
        # bound-check ROI
        x1 = max(cx - r, 0)
        x2 = min(cx + r, thresh.shape[1]-1)
        y1 = max(cy - r, 0)
        y2 = min(cy + r, thresh.shape[0]-1)
        roi = thresh[y1:y2, x1:x2]
        fill = cv2.countNonZero(roi)
        fills.append(fill)
        centers.append((cx,cy))
        cv2.circle(debug, (cx,cy), r, (0,0,255), 1)

    # Choose the option with largest fill
    marked_idx = int(np.argmax(fills))
    # Optionally detect blank/multi-mark: check top two closeness
    sorted_f = sorted(fills, reverse=True)
    ambiguous = False
    if sorted_f[0] < 5:  # nothing filled enough → blank
        detected = None
    elif len(sorted_f) > 1 and sorted_f[1] > 0 and sorted_f[0] - sorted_f[1] < max(3, 0.2*sorted_f[0]):
        # second is close to first → ambiguous / multi-mark
        detected = None
        ambiguous = True
    else:
        detected = chr(65 + marked_idx)  # 'A' + index

    answers[q] = {
        "detected": detected,
        "fills": fills,
        "ambiguous": ambiguous
    }
    if detected is not None:
        cv2.circle(debug, centers[marked_idx], r, (0,255,0), 2)
    else:
        # mark ambiguous/blank in yellow
        cv2.circle(debug, centers[marked_idx], r, (0,255,255), 2)

# ========== EVALUATION & SAVE CSV ==========
results = []
score = 0
for q in range(1, SUBJECTS*Q_PER_SUB + 1):
    detected = answers[q]["detected"]
    correct = answer_key.get(q)
    is_correct = (detected == correct)
    if is_correct:
        score += 1
    results.append([q, "" if detected is None else detected, "" if correct is None else correct, is_correct, answers[q]["ambiguous"]])

csv_path = "omr_result.csv"
with open(csv_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Question","Detected","Correct","IsCorrect","Ambiguous"])
    writer.writerows(results)

print(f"Final score: {score} / {len([k for k in answer_key if answer_key[k] is not None])}")
print(f"Results saved: {csv_path}")
cv2.imwrite("omr_debug.png", debug)
print("Debug image saved: omr_debug.png (green = chosen, yellow = blank/ambiguous)")

cv2.imshow("Detected Bubbles (debug)", debug)
cv2.waitKey(0)
cv2.destroyAllWindows()
