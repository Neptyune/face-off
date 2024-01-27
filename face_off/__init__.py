from flask import Flask, render_template, Response
import cv2
import numpy as np
from collections import deque
from flask_socketio import SocketIO
from deepface import DeepFace


app = Flask(__name__)
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

@app.route("/singleboard")
def singleboard():
    scores = {
        'Alice': 75.5,
        'Bob': 92.3,
        'Charlie': 88.0,
        'David': 65.8,
        'Eva': 78.9,
        'Annie': 100.0,
        'Nick': 10
    }
    sorted_scores = dict(sorted(scores.items(), key=lambda item: item[1], reverse=True))

    return render_template("singleboard.html", len = len(scores.keys()), names = list(sorted_scores.keys()), scores = list(sorted_scores.values()), emotion = "Happy")

@app.route("/video")
def video():
    return Response(
        generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame"
    )


emotion_history = {
    "angry": deque(maxlen=3),
    "disgust": deque(maxlen=3),
    "fear": deque(maxlen=3),
    "happy": deque(maxlen=3),
    "sad": deque(maxlen=3),
    "surprise": deque(maxlen=3),
    "neutral": deque(maxlen=3),
}


def update_emotions():
    frame_counter = 0
    while True:
        success, frame = camera.read()  # read the camera frame
        if not success:
            print("Camera not working!")
        else:
            # get the size of the frame
            # height, width, channels = frame.shape
            # print(height, width, channels)
            frame = frame[100:400, 200:500]
            # height, width, channels = frame.shape
            # print(height, width, channels)
            result = DeepFace.analyze(
                frame, enforce_detection=False, actions=["emotion"]
            )

            # for every json element in result[0]["emotion"], replace its value with the processed value
            # for emotion in result[0]["emotion"]:
            #     result[0]["emotion"][emotion] = process_raw_score(result[0]["emotion"][emotion])

            # for every json element in result[0]["emotion"], replace its value with the processed value
            for emotion in result[0]["emotion"]:
                processed_score = process_raw_score(result[0]["emotion"][emotion])
                emotion_history[emotion].append(processed_score)
                result[0]["emotion"][emotion] = sum(emotion_history[emotion]) / len(
                    emotion_history[emotion]
                )

            frame_counter += 1
            if frame_counter == 3:
                socketio.emit("update_emotion", {"emotion": result[0]["emotion"]})
                frame_counter = 0

        # socketio.sleep(1) # If enabled the number will lag behind the camera


def process_raw_score(x):
    scaling_factor = 0.1 / np.e
    return 100 - 100 * np.exp(-scaling_factor * x)


if __name__ == "__main__":
    try:
        socketio.start_background_task(update_emotions)
        socketio.run(app, debug=False, use_reloader=False, log_output=False)
    finally:
        camera.release()
        print("\nApplication exited and camera released")
