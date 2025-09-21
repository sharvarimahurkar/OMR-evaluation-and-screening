import cv2

# 1. Load your image
image_path = r"C:/Users/Sharvari Mahurkar/Downloads/Img1.jpeg"
image = cv2.imread(image_path)

if image is None:
    raise Exception("❌ Could not load image. Check the file path!")

# 2. Show image
cv2.imshow("Original OMR Sheet", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
