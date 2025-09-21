import cv2
import numpy as np

# Load image
image_path = r"C:/Users/Sharvari Mahurkar/Downloads/Img1.jpeg"
image = cv2.imread(image_path)

if image is None:
    raise Exception("❌ Could not load image. Check the path!")

# Resize for easier display (optional)
scale_percent = 50
width = int(image.shape[1] * scale_percent / 100)
height = int(image.shape[0] * scale_percent / 100)
dim = (width, height)
resized = cv2.resize(image, dim)

points = []

def get_points(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))
        cv2.circle(resized, (x, y), 5, (0, 0, 255), -1)
        cv2.imshow("Select 4 corners", resized)

cv2.imshow("Select 4 corners", resized)
cv2.setMouseCallback("Select 4 corners", get_points)
cv2.waitKey(0)
cv2.destroyAllWindows()

if len(points) == 4:
    # Scale back to original coordinates
    points = [(int(p[0] * 100 / scale_percent), int(p[1] * 100 / scale_percent)) for p in points]

    # Sort points automatically: top-left, top-right, bottom-right, bottom-left
    pts = np.array(points, dtype="float32")
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)

    top_left = pts[np.argmin(s)]
    bottom_right = pts[np.argmax(s)]
    top_right = pts[np.argmin(diff)]
    bottom_left = pts[np.argmax(diff)]

    pts1 = np.float32([top_left, top_right, bottom_right, bottom_left])

    # Output size (keep closer to A4 ratio)
    width, height = 800, 1100
    pts2 = np.float32([[0,0], [width,0], [width,height], [0,height]])

    # Warp
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    warped = cv2.warpPerspective(image, matrix, (width, height))

    # Save and show
    cv2.imwrite("omr_cropped.jpg", warped)
    cv2.imshow("Warped OMR", warped)
    print("✅ Warped image saved as omr_cropped.jpg")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("❌ Please select exactly 4 corners")
