import cv2
from deepface import DeepFace

# Start webcam
cap = cv2.VideoCapture(0)

# Check if webcam is opened correctly
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Set a smaller frame size for faster processing
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    try:
        # Convert to RGB as DeepFace expects RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Analyze the frame using DeepFace - simplified parameters
        result = DeepFace.analyze(
            img_path=rgb_frame,
            actions=['emotion'],
            enforce_detection=False,  # Continue even if no face detected
            detector_backend='opencv',  # Faster face detection
            silent=True  # Disable progress messages
        )

        # Check if any face was detected
        if isinstance(result, list) and len(result) > 0:
            # Extract emotion and face location from first face found
            dominant_emotion = result[0]['dominant_emotion']
            region = result[0]['region']
            x, y, w, h = region['x'], region['y'], region['w'], region['h']

            # Draw bounding box and emotion label
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f"Emotion: {dominant_emotion}", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Display confidence scores for debugging
            emotions = result[0]['emotion']
            emotion_text = ", ".join([f"{k}:{v:.1f}%" for k, v in emotions.items()])
            cv2.putText(frame, emotion_text, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

    except Exception as e:
        # Display "No face detected" message
        cv2.putText(frame, "No face detected", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.1, (0, 0, 255), 1)

    # Show the frame
    cv2.imshow("Emotion", frame)

    # Quit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()