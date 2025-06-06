import cv2
import time


def facerec():
    """A real-time face detector which uses openCV and haarcascade classifier to detect
    faces that the trainer model has already been trained on."""

    # Load the trained model for face detection and recognition
    face_detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    face_recognizer = cv2.face.LBPHFaceRecognizer_create()
    face_recognizer.read("trainer/trainer.yml")

    names = {1: "Person 1", 2: "Person 2", 3: "Person 3", 4: "Person 4"}
    # names = ["None", "Huzaifa", "Somebody Else"]

    # Set up the video capture
    cap = cv2.VideoCapture(0)

    lock_timer = None

    # Load the images of the faces you want to recognize
    face_images = {}
    for i in range(1, 11):
        face_images[i] = cv2.imread(f"face{i}.jpg", cv2.IMREAD_GRAYSCALE)

    while True:
        # Capture and process a frame from the video stream
        ret, frame = cap.read()
        if not ret:
            break

        # Convert the frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Use the face detector to detect faces in the frame
        faces = face_detector.detectMultiScale(gray, 1.3, 5)
        detected = False
        # Recognize the detected faces
        for (x, y, w, h) in faces:
            face_roi = gray[y:y + h, x:x + w]
            # label, confidence = face_recognizer.predict(face_roi)
            label, confidence = face_recognizer.predict(face_roi)
            if confidence < 100:
                name = f"face{label}"
                # name = names[label]
                print("Unlock")
            else:
                name = "Unknown"


            # Draw a rectangle and label around the recognized face
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(frame, name, (x, y - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        # if doorUnlock == True and time.time() - prevTime > 10:
        #     doorUnlock = False
        #     print("door lock")


        # Display the resulting frame
        cv2.imshow('Video', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        time.sleep(5)
        print("Lock")

    # Release the video capture and close the window
    cap.release()
    cv2.destroyAllWindows()


facerec()
