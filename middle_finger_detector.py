import cv2
import mediapipe as mp
import numpy as np
import os
import platform
import time

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.3,
    min_tracking_confidence=0.3
)
mp_drawing = mp.solutions.drawing_utils

# Initialize webcam
cap = cv2.VideoCapture(0)

def shutdown_system():
    print("Shutdown will commence in 0.5 seconds...")
    time.sleep(0.2)  # Add half second delay
    if platform.system() == 'Linux':
        os.system('shutdown now')
    elif platform.system() == 'Darwin':  # macOS
        os.system('shutdown -h now')
    elif platform.system() == 'Windows':
        os.system('shutdown /s /t 0')

def is_middle_finger_raised(hand_landmarks):
    # Get the y-coordinates of the fingertips and their corresponding MCP (metacarpophalangeal) joints
    middle_tip = hand_landmarks.landmark[12].y
    middle_mcp = hand_landmarks.landmark[9].y
    
    # For other fingers: tips are 8, 16, 20; MCPs are 5, 13, 17
    index_tip = hand_landmarks.landmark[8].y
    index_mcp = hand_landmarks.landmark[5].y
    
    ring_tip = hand_landmarks.landmark[16].y
    ring_mcp = hand_landmarks.landmark[13].y
    
    pinky_tip = hand_landmarks.landmark[20].y
    pinky_mcp = hand_landmarks.landmark[17].y
    
    # Check if middle finger is significantly raised (tip is well above MCP)
    middle_raised = (middle_tip < middle_mcp - 0.1) 
    
    # Check for other fingers being down (more lenient)
    # Each finger just needs to be slightly below its MCP joint
    index_down = (index_tip > index_mcp - 0.02)  
    ring_down = (ring_tip > ring_mcp - 0.02)    
    pinky_down = (pinky_tip > pinky_mcp - 0.02)
    
    # Calculate the difference to ensure middle finger is raised
    middle_diff = middle_mcp - middle_tip
    threshold = 0.1  
    
    return middle_raised and index_down and ring_down and pinky_down and (middle_diff > threshold)

last_alert_time = 0
alert_cooldown = 0.5
shutdown_initiated = False

print("Starting middle finger detection...")
print("WARNING: Middle finger gesture will initiate system shutdown!")
print("Make sure your hand is visible to the camera")
print("Press 'q' to quit")

while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("Failed to grab frame")
        continue

    image = cv2.flip(image, 1)
    
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    results = hands.process(image_rgb)
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            if is_middle_finger_raised(hand_landmarks):
                current_time = time.time()
                if current_time - last_alert_time > alert_cooldown:
                    print("\n" + "="*50)
                    print("🚨 MIDDLE FINGER DETECTED! 🚨")
                    print("="*50 + "\n")
                    
                    # Draw a red circle around the middle finger tip
                    middle_tip = hand_landmarks.landmark[12]
                    x = int(middle_tip.x * image.shape[1])
                    y = int(middle_tip.y * image.shape[0])
                    cv2.circle(image, (x, y), 20, (0, 0, 255), 2)
                    cv2.putText(image, "SHUTDOWN INITIATED!", (x-100, y-30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    
                    if not shutdown_initiated:
                        shutdown_initiated = True
                        shutdown_system()
                    
                    last_alert_time = current_time
    
    cv2.imshow('Finger Motion Detection', image)
    
    # Break the loop if 'q' is pressed
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows() 
