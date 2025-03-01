import torch
import os
import soundfile as sf
from datetime import datetime
from flask import Flask, request, jsonify, send_file
from tts.app import tts

app = Flask(__name__)
model_name = "U.E.P_normal"

@app.route('/synthesize', methods=['POST'])
def synthesize():
    data = request.json
    text = data.get("text", "")
    f0_up_key = data.get("f0_up_key", 0)
    f0_method = data.get("f0_method", "rmvpe")  # Default pitch extraction method
    index_rate = data.get("index_rate", 0)
    protect = data.get("protect", 0.33)

    if not text:
        return jsonify({"error": "Text input is required"}), 400
    
    print("\nAccepted request from client: ", request.remote_addr)
    print("Current time: ", datetime.now())

    # Run text through tts pipeline and generate audio
    try:
        print("Starting TTS synthesis...\n--------------------------------\nArguments as below:")
        print("Text:", text)
        print("F0 Up Key:", f0_up_key)
        print("F0 Method:", f0_method)
        print("Index Rate:", index_rate)
        print("Protect:", protect)
        print("--------------------------------")
        
        _, _, audio_data = tts(
            model_name, text, f0_up_key, f0_method, index_rate, protect
        )
        
        output_path = os.path.join(os.getcwd(), "tts", 'output.wav')  # 指定要保存的檔案名稱和路徑
        sf.write(output_path, audio_data[1], audio_data[0])
        
        try:
            if audio_data:
                audio_path = output_path
                print("Audio file successfully generated, sending the file to remote client...\n")
                return send_file(audio_path, mimetype="audio/wav")
            else:
                return jsonify({"error": "Synthesis failed."}), 500
            
        except Exception as e:
            print("Error processing audio synthesize:", e)
            return jsonify({"error": str(e)}), 500
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500
        
class server:
    def __init__(self):
        print(torch.cuda.is_available())
        self.ip = '26.87.187.124'
        self.port = 5000

    def call(self):
        app.run(host=self.ip, port=self.port)
    
if __name__ == '__main__':
    instance = server()
    instance.call()