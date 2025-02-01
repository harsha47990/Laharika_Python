import cv2

# Callback function to capture mouse click events
def get_coordinates(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:  # Left mouse button click
        print(f"Clicked at: X={x}, Y={y}")

# Load an image (Change the path to your image file)
image_path = r"C:\Users\harsha.martha\OneDrive - WIRB-Copernicus Group, Inc\Documents\harsha\id\front.png"
image = cv2.imread(image_path)

# Create a window and set the mouse callback function
cv2.namedWindow("Image Viewer")
cv2.setMouseCallback("Image Viewer", get_coordinates)

# Display the image
while True:
    cv2.imshow("Image Viewer", image)
    if cv2.waitKey(1) & 0xFF == 27:  # Press 'ESC' to exit
        break

cv2.destroyAllWindows()
