import os

from flask import Flask, request, jsonify, render_template

from modules.preprocessing import preprocess
from modules.processing import processing
from modules.voice_to_text import transcribe_audio, save_transcription

app = Flask(__name__)
app.config["DEBUG"] = True


@app.route("/")
def hello():
    return render_template("pages/index.html")


@app.route('/upload-audio', methods=['POST'])
def upload_audio():
    try:
        if 'audio' not in request.files:
            return jsonify({'error': 'No audio file provided'}), 400

        audio_file = request.files['audio']
        audioName = f"static\\{audio_file.filename}"
        audio_path = os.path.join(os.getcwd(), audioName)
        audio_file.save(audio_path)
        transcription = transcribe_audio(audio_path)

        save_transcript_name = "temp\\transcription.xlsx"
        save_transcript_path = os.path.join(os.getcwd(), save_transcript_name)
        save_transcription(transcription, save_transcript_path)

        preprocess_df = preprocess(save_transcript_path)

        result = processing()

        return jsonify({'response': transcription.text, 'preprocess': preprocess_df.to_json(), 'result': result}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500
