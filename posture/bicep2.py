import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe pose model
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Function to calculate joint angle
def calculate_angle(a, b, c):
    a = np.array(a)  # Shoulder
    b = np.array(b)  # Elbow
    c = np.array(c)  # Wrist
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0:
        angle = 360 - angle
    return angle

# Function to process image and detect posture
def analyze_bicep_curl(image_path):
    # Read the image
    image = cv2.imread("/Users/bizzlemuffinn/Downloads/posture_correction-main/shutterstock_657941434_376ae0c9-1a39-42d3-bc39-3eaf18d5038f_1000x.jpg.webp")
    if image is None:
        print("Error: Could not load the image")
        return
    
    # Convert BGR image to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Initialize MediaPipe Pose
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        # Process the image
        results = pose.process(image_rgb)
        
        # Convert back to BGR for OpenCV
        image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        
        try:
            # Extract key landmarks
            landmarks = results.pose_landmarks.landmark
            
            # Get coordinates for the left arm
            shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                       landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                    landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
            wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                    landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
            
            # Convert to pixel coordinates
            h, w, _ = image.shape
            shoulder = (int(shoulder[0] * w), int(shoulder[1] * h))
            elbow = (int(elbow[0] * w), int(elbow[1] * h))
            wrist = (int(wrist[0] * w), int(wrist[1] * h))
            
            # Calculate angle at elbow
            angle = calculate_angle(shoulder, elbow, wrist)
            
            # Draw lines and points on the arm
            cv2.circle(image, shoulder, 8, (255, 0, 0), -1)  # Shoulder (Blue)
            cv2.circle(image, elbow, 8, (0, 255, 0), -1)     # Elbow (Green)
            cv2.circle(image, wrist, 8, (0, 0, 255), -1)     # Wrist (Red)
            cv2.line(image, shoulder, elbow, (255, 255, 255), 2)
            cv2.line(image, elbow, wrist, (255, 255, 255), 2)
            
            # Provide feedback on the bicep curl form
            if angle < 90:
                cv2.putText(image, "Good Curl stop!", (50, 50),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            else:
                cv2.putText(image, "Keep curling", (50, 50),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                
        except Exception as e:
            print(f"No pose detected or error occurred: {str(e)}")
            cv2.putText(image, "Pose not detected", (50, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # Display the output
        cv2.imshow('Bicep Curl Analysis', image)
        cv2.waitKey(0)  # Wait for any key press to close
        cv2.destroyAllWindows()

# Example usage
if __name__ == "__main__":
    # Replace 'path_to_your_image.jpg' with the actual path to your image
    image_path = 'path_to_your_image.jpg'
    analyze_bicep_curl(image_path)