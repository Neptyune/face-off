import os
from cv2.typing import MatLike
from flask import Flask, json, render_template, Response
from flask_socketio import SocketIO
from typing import Dict, TypeAlias
from deepface import DeepFace
import cv2
import matplotlib.pyplot as plt
from time import time

IMAGES_PATH = os.path.join(os.getcwd(), "images")
print(IMAGES_PATH)
EMOTIONS = ("angry", "disgust", "fear", "happy", "sad", "surprise", "neutral")


def save_frame(frame: MatLike, emotion: str, score: float):
    """Saves the given frame"""
    file_name = os.path.join(IMAGES_PATH, f"{emotion}-{round(score, 8)}.jpg")
    print(frame)
    plt.imshow(frame)
    plt.show()
    print("Save result:", cv2.imwrite(file_name, frame))
    return file_name


def remove_frame(file_name: str):
    try:
        os.remove(os.path.join(IMAGES_PATH, file_name))
    except FileNotFoundError:
        print("WARN - File name miss-match, frame was not removed")


EmotionScores: TypeAlias = Dict[
    str, str
]  # Reality its Dict[float, str] but json makes it a string


class Leaderboard:
    """Rate limited leaderboard functions"""

    def __init__(self, leaderboard_file: str) -> None:
        self.leaderboard_file = leaderboard_file
        self.leaderboard: Dict[str, EmotionScores] = {}
        self.last_write = 0

    def generate_leaderboard_file(self):
        emotion_scores = {}
        for emotion in EMOTIONS:
            emotion_scores[emotion] = {}
        with open(self.leaderboard_file, "w") as fp:
            json.dump(emotion_scores, fp, indent=4)

    def load_leaderboard(self):
        with open(self.leaderboard_file, "r") as fp:
            self.leaderboard = json.load(fp)

    def save_leaderboard(self):
        with open(self.leaderboard_file, "w") as fp:
            json.dump(self.leaderboard, fp, indent=4)

    def handle_new_score(self, frame: MatLike, emotion: str, new_score: float):
        emotion_scores: EmotionScores = self.leaderboard[emotion]

        lowest_score = min(map(float, emotion_scores.keys())) if emotion_scores else 0.0
        if new_score >= lowest_score and time() - self.last_write >= 0.5:
            self.last_write = time()
            file_name = save_frame(frame, emotion, new_score)
            emotion_scores[str(new_score)] = file_name

        # Cleanup and remove old score's frame
        if len(emotion_scores.keys()) > 3:
            lowest_score = str(lowest_score)
            remove_frame(emotion_scores[lowest_score])
            emotion_scores.pop(lowest_score)


app = Flask(__name__)
# app.config["SECRET_KEY"] = "secret!"
socketio = SocketIO(app)

camera = cv2.VideoCapture(0)
if not camera.isOpened():
    print("Unable to open camera")

leaderboard = Leaderboard("leaderboard.json")
# leaderboard.generate_leaderboard_file()
leaderboard.load_leaderboard()


@app.route("/")
def index():
    return render_template("index.html")  # return index.html page


# A leader board page with the top 10 scores
@app.route("/leaderboard")
def leaderboard_func():
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
            )[0]
            if len(result):
                emotion = result["dominant_emotion"]
                leaderboard.handle_new_score(
                    frame,
                    emotion,
                    result["emotion"][emotion],
                )
                socketio.emit("update_emotion", {"emotion": result["emotion"]})

        # socketio.sleep(1) # If enabled the number will lag behind the camera


if __name__ == "__main__":
    try:
        socketio.start_background_task(update_emotions)
        socketio.run(app, debug=False, use_reloader=False, log_output=False)
    finally:
        camera.release()
        leaderboard.save_leaderboard()
        print("\nApplication exited and camera released")


# class Emotion(Enum):
#     ANGRY = "angry"
#     DISGUST = "disgust"
#     FEAR = "fear"
#     HAPPY = "happy"
#     SAD = "sad"
#     SURPRISE = "surprise"
#     NEUTRAL = "neutral"
