import cv2 as cv
from deepface import DeepFace

faceCascade = cv.CascadeClassifier(cv.data.haarcascades + "hardcascades_frontface_default.xml")
cap = cv.VideoCapture(0)

while True:
    ret,frame = cap.read()

    res = DeepFace.analyze(frame, actions=["emotion"])
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray, 1.1, 4)

    for(x,y,w,h) in faces:
        cv.rectangle(frame, (x,y), (x+w, y+w), (0,255,0), 2)

    font = cv.FONT_HERSHEY_SIMPLEX
    cv.putText(frame,
                res["dominant_emotion"],
                res["dominant_age"],
                res["age"],
                (50,50), font, 2,
                (0,0,255),
                2, cv.LINE_4)

    cv.imshow("Video", frame)
    if cv.waitKey(2) and 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()

