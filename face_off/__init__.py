from flask import Flask, render_template, Response
import cv2
from flask_socketio import SocketIO
from deepface import DeepFace


app = Flask(__name__)
# app.config["SECRET_KEY"] = "secret!"
socketio = SocketIO(app)

camera = cv2.VideoCapture(0)
if not camera.isOpened():
    print("Unable to open camera")


@app.route("/")
def index():
    return render_template("index.html")  # return index.html page


# A leader board page with the top 10 scores
@app.route("/leaderboard")
def leaderboard():
    return render_template("leaderboard.html")


def generate_frames():
    while True:
        success, frame = camera.read()  # read the camera frame
        if not success:
            break
        else:
            ret, buffer = cv2.imencode(".jpg", frame)
            frame = buffer.tobytes()
            yield (
                b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
            )  # concat frame one by one and show result


@app.route("/video")
def video():
    return Response(
        generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame"
    )


def update_emotions():
    while True:
        success, frame = camera.read()  # read the camera frame
        if not success:
            print("Camera not working!")
        else:
            result = DeepFace.analyze(
                frame, enforce_detection=False, actions=["emotion"]
            )
            socketio.emit("update_emotion", {"emotion": result[0]["emotion"]})

        # socketio.sleep(1) # If enabled the number will lag behind the camera


if __name__ == "__main__":
    try:
        socketio.start_background_task(update_emotions)
        socketio.run(app, debug=False, use_reloader=False, log_output=False)
    finally:
        camera.release()
        print("\nApplication exited and camera released")
