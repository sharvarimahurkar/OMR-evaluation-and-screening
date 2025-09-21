import cv2

# Load the warped OMR sheet (from Step 2)
image_path = "omr_cropped.jpg"
warped = cv2.imread(image_path)

if warped is None:
    raise Exception("❌ Could not load warped image. Run Step 2 first!")

# List to store calibration clicks
calib_points = []

def select_bubbles(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        calib_points.append((x, y))
        cv2.circle(warped_display, (x,y), 5, (255,0,0), -1)
        cv2.imshow("Calibration", warped_display)

warped_display = warped.copy()
cv2.imshow("Calibration", warped_display)
print("👉 Click 4 bubbles in this order:\n"
      "1. Q1 → Option A\n"
      "2. Q1 → Option B\n"
      "3. Q2 → Option A\n"
      "4. Q21 → Option A (first bubble in next subject column)")
cv2.setMouseCallback("Calibration", select_bubbles)
cv2.waitKey(0)
cv2.destroyAllWindows()

if len(calib_points) != 4:
    raise Exception("❌ You must click exactly 4 calibration bubbles!")

# Extract calibration points
Q1A, Q1B, Q2A, Q21A = calib_points

# Calculate bubble grid spacing
x_gap = abs(Q1B[0] - Q1A[0])   # horizontal gap between A-B
y_gap = abs(Q2A[1] - Q1A[1])   # vertical gap between Q1-Q2
col_offset = abs(Q21A[0] - Q1A[0])  # horizontal offset for next subject block

print("✅ Calibration complete!")
print(f"Horizontal gap (x_gap): {x_gap}")
print(f"Vertical gap (y_gap): {y_gap}")
print(f"Column offset (col_offset): {col_offset}")
