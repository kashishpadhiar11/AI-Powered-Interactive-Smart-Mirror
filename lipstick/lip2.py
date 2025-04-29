import cv2
import dlib
import numpy as np
import os
import urllib.request
import bz2

# Define file names and URL for the shape predictor
PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"
PREDICTOR_URL = "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2"

def download_progress_hook(count, block_size, total_size):
    """Prints the download progress percentage."""
    percent = int(count * block_size * 100 / total_size)
    print(f"\rDownloading: {percent}%", end="", flush=True)

# Download and extract the shape predictor file if it does not exist
if not os.path.exists(PREDICTOR_PATH):
    print(f"'{PREDICTOR_PATH}' not found. Downloading from {PREDICTOR_URL} ...")
    compressed_file = PREDICTOR_PATH + ".bz2"
    try:
        urllib.request.urlretrieve(PREDICTOR_URL, compressed_file, reporthook=download_progress_hook)
        print("\nDownload complete. Extracting the file...")
        with bz2.BZ2File(compressed_file, 'rb') as file, open(PREDICTOR_PATH, 'wb') as out_file:
            out_file.write(file.read())
        os.remove(compressed_file)
        print("Extraction complete. Proceeding with the application...")
    except Exception as e:
        raise RuntimeError("Failed to download and extract the shape predictor file.") from e

# Define a dictionary of available color options in BGR format
color_options = {
    "red":     (0, 0, 255),
    "green":   (0, 255, 0),
    "blue":    (255, 0, 0),
    "yellow":  (0, 255, 255),
    "cyan":    (255, 255, 0),
    "magenta": (255, 0, 255),
    "white":   (255, 255, 255),
    "black":   (0, 0, 0),
    "orange":  (0, 165, 255),
    "purple":  (128, 0, 128)
}

# Display available color options and prompt the user for a choice
print("\nAvailable Lipstick Colors:")
for name in color_options:
    print(f" - {name.title()}")

choice = input("\nEnter your choice of color: ").strip().lower()
if choice in color_options:
    LIPSTICK_COLOR = color_options[choice]
    print(f"Using {choice.title()} lipstick!")
else:
    print("Color not recognized. Defaulting to Red.")
    LIPSTICK_COLOR = color_options["red"]

# Other constants for the lipstick application
SCALE_FACTOR = 1        # Controls image scaling for processing
FEATHER_AMOUNT = 11     # Controls edge feathering for blending
OPACITY = 0.4           # Opacity of the lipstick (0.0 to 1.0)

# Load face detector and landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(PREDICTOR_PATH)

def get_landmarks(im):
    rects = detector(im, 0)
    if len(rects) == 0:
        return None
    return np.matrix([[p.x, p.y] for p in predictor(im, rects[0]).parts()])

def draw_convex_hull(im, points, color):
    points = cv2.convexHull(points)
    cv2.fillConvexPoly(im, points, color=color)

def get_lip_mask(im, landmarks):
    # The mouth landmarks are points 48 to 67
    lip_points = landmarks[48:68]
    mask = np.zeros(im.shape[:2], dtype=np.float32)
    draw_convex_hull(mask, lip_points, color=1)
    # Convert the single-channel mask to three channels
    mask = np.array([mask, mask, mask]).transpose((1, 2, 0))
    # Feather the mask edges
    mask = (cv2.GaussianBlur(mask, (FEATHER_AMOUNT, FEATHER_AMOUNT), 0) > 0) * 1.0
    return cv2.GaussianBlur(mask, (FEATHER_AMOUNT, FEATHER_AMOUNT), 0)

def apply_lipstick(im, mask, color, opacity):
    # Create an image filled with the chosen lipstick color
    color_image = np.zeros(im.shape, dtype=im.dtype)
    color_image[:] = color
    # Multiply the mask with the color image
    color_mask = (mask * color_image).astype(im.dtype)
    # Blend the color mask with the original image
    return cv2.addWeighted(color_mask, opacity, im, 1 - opacity, 0)

# Get image path from user
image_path = input("\nPlease enter the path to your image: ").strip()

# Read and process the image
try:
    frame = cv2.imread(image_path)
    if frame is None:
        raise ValueError("Could not load the image. Please check the file path.")
    
    # Create a resizable window
    cv2.namedWindow('Lipstick Try-On', cv2.WINDOW_NORMAL)
    
    # Process the image
    small_frame = cv2.resize(frame, (0, 0), fx=1/SCALE_FACTOR, fy=1/SCALE_FACTOR)
    gray_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
    landmarks = get_landmarks(gray_frame)

    if landmarks is not None:
        landmarks = landmarks * SCALE_FACTOR
        lip_mask = get_lip_mask(frame, landmarks)
        output = apply_lipstick(frame, lip_mask, LIPSTICK_COLOR, OPACITY)
    else:
        output = frame
        print("No face detected in the image.")

    # Display the result
    cv2.imshow('Lipstick Try-On', output)
    
    # Save the output
    output_path = "output_with_lipstick.jpg"
    cv2.imwrite(output_path, output)
    print(f"Output saved as '{output_path}'")
    
    # Wait for key press and then close
    cv2.waitKey(0)
    cv2.destroyAllWindows()

except Exception as e:
    print(f"Error: {str(e)}")