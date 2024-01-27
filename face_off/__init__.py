from flask import Flask, render_template, Response
import cv2

app = Flask(__name__)


def generate_frames():
    camera = cv2.VideoCapture(0)
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


@app.route("/")
def index():
    return render_template("index.html")  # return index.html page

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


if __name__ == "__main__":
    app.run(debug=True)
