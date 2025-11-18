import cv2
from deepface import DeepFace
import pyttsx3

# Initialize text-to-speech engine
engine = pyttsx3.init()
engine.setProperty('rate', 150)  # Speed of speech
engine.setProperty('volume', 1.0)

# Function to speak
def speak(text):
    engine.say(text)
    engine.runAndWait()

# Start webcam
cap = cv2.VideoCapture(0)

last_emotion = ""  # To avoid repeating voice continuously

while True:
    key, img = cap.read()

    # Analyze emotion
    results = DeepFace.analyze(img, actions=['emotion'], enforce_detection=False)
    emotion = results[0]['dominant_emotion']

    # Display emotion on screen
    cv2.putText(img, f'Emotion: {emotion}', (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    # Speak only when emotion changes
    if emotion != last_emotion:
        speak(f"The detected emotion is {emotion}")
        last_emotion = emotion

    cv2.imshow("Emotion Recognition", img)

    # Exit when pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
