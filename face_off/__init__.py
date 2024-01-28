import os
from cv2.typing import MatLike
from flask import Flask, json, render_template, Response
import numpy as np
from collections import deque
from flask_socketio import SocketIO
from typing import Dict, Tuple, TypeAlias
from deepface import DeepFace
import cv2
from time import time


IMAGES_PATH = "images"
EMOTIONS = ("angry", "disgust", "fear", "happy", "sad", "surprise", "neutral")


def save_frame(frame: MatLike, emotion: str, score: float):
    """Saves the given frame"""
    file_name = os.path.join(IMAGES_PATH, f"{emotion}-{round(score, 8)}.jpg")

    cv2.imwrite(file_name, frame)
    return file_name


def remove_frame(file_name: str):
    try:
        os.remove(os.path.join(IMAGES_PATH, file_name))
    except FileNotFoundError:
        pass
        # print("WARN - File name miss-match, frame was not removed")


EmotionScores: TypeAlias = Dict[
    str, Tuple[str, str]
]  # Reality its Dict[float, str] but json makes it a string


class Tracker:
    def __init__(self) -> None:
        self.scores_and_frames: Dict[str, Tuple[float, MatLike]] = {}

    def handle_new_score(self, frame: MatLike, emotion: str, new_score: float):
        try:
            if new_score > self.scores_and_frames[emotion][0]:
                self.scores_and_frames[emotion] = (new_score, frame)
        except KeyError:
            self.scores_and_frames[emotion] = (new_score, frame)

    def clear(self):
        self.scores_and_frames = {}


class Leaderboard:
    """Rate limited leaderboard functions"""

    def __init__(self, leaderboard_file: str) -> None:
        self.leaderboard_file = leaderboard_file
        self.leaderboard: Dict[str, EmotionScores] = {}
        self.last_write = 0

    def generate_file(self):
        emotion_scores = {}
        for emotion in EMOTIONS:
            emotion_scores[emotion] = {}
        with open(self.leaderboard_file, "w") as fp:
            json.dump(emotion_scores, fp, indent=4)

    def load(self):
        with open(self.leaderboard_file, "r") as fp:
            self.leaderboard = json.load(fp)

    def update_and_save(self, tracker: Tracker, name: str):
        new_highscore = False
        for emotion in tracker.scores_and_frames:
            emotion_scores: EmotionScores = self.leaderboard[emotion]
            tracked_emotion = tracker.scores_and_frames[emotion]  # (score, frame)

            lowest_score = (
                min(map(float, emotion_scores.keys())) if emotion_scores else 0.0
            )
            if tracked_emotion[0] >= lowest_score and time() - self.last_write >= 0.5:
                file_name = save_frame(tracked_emotion[1], emotion, tracked_emotion[0])
                emotion_scores[str(tracked_emotion[0])] = (file_name, name)
                new_highscore = True

            # Cleanup and remove old score's frame
            if len(emotion_scores.keys()) > 3:
                lowest_score = str(lowest_score)
                remove_frame(emotion_scores[lowest_score][0])
                emotion_scores.pop(lowest_score)

        if new_highscore:
            with open(self.leaderboard_file, "w") as fp:
                json.dump(self.leaderboard, fp, indent=4)

        return new_highscore


app = Flask(__name__)
socketio = SocketIO(app)

camera = cv2.VideoCapture(0)
if not camera.isOpened():
    print("Unable to open camera")

leaderboard = Leaderboard("leaderboard.json")
# leaderboard.generate_file()
leaderboard.load()

tracker = Tracker()


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


@app.route("/singleboard")
def singleboard():
    scores = {
        "Alice": 75.5,
        "Bob": 92.3,
        "Charlie": 88.0,
        "David": 65.8,
        "Eva": 78.9,
        "Annie": 100.0,
        "Nick": 10,
    }
    sorted_scores = dict(sorted(scores.items(), key=lambda item: item[1], reverse=True))

    return render_template(
        "singleboard.html",
        len=len(scores.keys()),
        names=list(sorted_scores.keys()),
        scores=list(sorted_scores.values()),
        emotion="Happy",
    )


@app.route("/video")
def video():
    return Response(
        generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame"
    )


def update_emotions():
    emotion_history = {
        "angry": deque(maxlen=3),
        "disgust": deque(maxlen=3),
        "fear": deque(maxlen=3),
        "happy": deque(maxlen=3),
        "sad": deque(maxlen=3),
        "surprise": deque(maxlen=3),
        "neutral": deque(maxlen=3),
    }
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
            )[0]
            if len(result):
                emotion = result["dominant_emotion"]

                processed_emotions, emotion_history = process_emotions(
                    result["emotion"], emotion_history
                )

                tracker.handle_new_score(
                    frame,
                    emotion,
                    processed_emotions[emotion],
                )

                socketio.emit("update_emotion", {"emotion": processed_emotions})

        # socketio.sleep(1) # If enabled the number will lag behind the camera


def process_emotions(emotion_dict: Dict[str, float], emotion_history):
    """Processes the scores to use the average of the last 3 data points along with a exponential function"""
    processed_emotion_dict = {}
    for emotion in emotion_dict:
        processed_score = process_raw_score(emotion_dict[emotion])
        emotion_history[emotion].append(processed_score)
        processed_emotion_dict[emotion] = sum(emotion_history[emotion]) / len(
            emotion_history[emotion]
        )
    return processed_emotion_dict, emotion_history


def process_raw_score(x):
    scaling_factor = 0.1 / np.e
    return 100 - 100 * np.exp(-scaling_factor * x)


@socketio.on("save")
def save_name_scores(data):
    print(f"New Highscore for {data['data']}:", leaderboard.update_and_save(tracker, data["data"]))
    tracker.clear()


if __name__ == "__main__":
    try:
        socketio.start_background_task(update_emotions)
        socketio.run(app, debug=False, use_reloader=False, log_output=False)
    finally:
        camera.release()
        print("\nCamera released, saving leaderboard.")


# class Emotion(Enum):
#     ANGRY = "angry"
#     DISGUST = "disgust"
#     FEAR = "fear"
#     HAPPY = "happy"
#     SAD = "sad"
#     SURPRISE = "surprise"
#     NEUTRAL = "neutral"
