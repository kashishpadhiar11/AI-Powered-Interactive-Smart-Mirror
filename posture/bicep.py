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

# Start webcam capture
cap = cv2.VideoCapture(0)

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert BGR image to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image)

        # Convert back to BGR for OpenCV
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

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
            cv2.circle(image, elbow, 8, (0, 255, 0), -1)      # Elbow (Green)
            cv2.circle(image, wrist, 8, (0, 0, 255), -1)      # Wrist (Red)
            cv2.line(image, shoulder, elbow, (255, 255, 255), 2)
            cv2.line(image, elbow, wrist, (255, 255, 255), 2)

            # Provide feedback on the bicep curl form
            if angle < 80:
                cv2.putText(image, "Good Curl stop!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                            1, (0, 255, 0), 2)
            else:
                cv2.putText(image, "Keep curling", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                            1, (0, 0, 255), 2)

        except:
            pass
        
        # Display the output frame
        cv2.imshow('Bicep Curl Tracker', image)

        # Exit on pressing 'q'
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

# Release resources
cap.release()
cv2.destroyAllWindows()
