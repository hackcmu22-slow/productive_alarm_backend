import cv2 
from flask import Flask, Response
from facerec import SimpleFacerec

# Encode faces from a folder
sfr = SimpleFacerec()
sfr.load_encoding_images("images/")

# Load Camera
cap = cv2.VideoCapture("http://localhost:8080/stream?topic=/argus/ar0234_front_left/scaled/image_raw&quality=15")

app = Flask(__name__)

@app.route('/feed')
def feed():
    def gen():
        while True:
            ret, frame = cap.read()

            # Detect Faces
            face_locations, face_names = sfr.detect_known_faces(frame)
            for face_loc, name in zip(face_locations, face_names):
                y1, x2, y2, x1 = face_loc[0], face_loc[1], face_loc[2], face_loc[3]

                cv2.putText(frame, name,(x1, y1 - 10), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 200), 2)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 200), 1)
                # print(cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 200), 1)
                print(f'{name} face detected')
            _, img = cv2.imencode('.jpg', frame)
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + img.tobytes() + b'\r\n')


    return Response(gen(), mimetype='multipart/x-mixed-replace;boundary=frame')

    # key = cv2.waitKey(1)
    # if key == 27:
    #     break
    # Hit 'q' on the keyboard to quit!
    #if cv2.waitKey(1) & 0xFF == ord('q'):
    #        break

