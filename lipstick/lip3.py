import cv2
import dlib
import numpy as np

# Load the pre-trained face detector and facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

def apply_lipstick(image_path, lipstick_color=(0, 0, 255)):
    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Unable to load image at {image_path}")
        return

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect faces in the image
    faces = detector(gray)
    if len(faces) == 0:
        print("No faces detected.")
        return

    for face in faces:
        # Get the landmarks
        landmarks = predictor(gray, face)

        # Extract lip coordinates
        lip_points = []
        for i in range(48, 61):  # Points for the outer and inner lips
            lip_points.append((landmarks.part(i).x, landmarks.part(i).y))

        # Create a mask for the lips
        lip_mask = np.zeros_like(image)
        cv2.fillPoly(lip_mask, [np.array(lip_points, dtype=np.int32)], lipstick_color)

        # Blend the lipstick color with the original image
        alpha = 0.6  # Adjust the intensity of the lipstick
        cv2.addWeighted(lip_mask, alpha, image, 1 - alpha, 0, image)

    # Display the result
    cv2.imshow("Virtual Lipstick", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Example usage
apply_lipstick("/Users/bizzlemuffinn/Desktop/lipstick/istockphoto-1442556244-612x612.jpg", lipstick_color=(0, 0, 255))  # Red lipstick