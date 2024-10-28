import torch
import numpy as np
import soundfile as sf
import os
from flask import Flask, request, jsonify

class server:
    def __init__(self):
        print(torch.cuda.is_available())
        self.app = Flask(__name__)
        model_path = os.path.join(os.path.dirname(__file__), 'U.E.P_sweet.pth')
        self.model = torch.load(model_path)
        
    def synthesize_speech(self, text):
        generated_audio = np.random.randn(16000 * 3)
        sf.write('output.wav', generated_audio, 16000)

        with open('output.wav', 'rb') as f:
            audio_byte = f.read()

        return audio_byte

    def synthesize(self):
        try:
            data = request.json
            text = data.get('text', '')

            if not text:
                return jsonify({'error': 'Text input is required'}), 400

            audio_data = self.synthesize_speech(text)

            return audio_data, 200, {
                'Content-Type': 'audio/wav',
                'Content-Disposition': 'attachment; filename=output.wav'
            }
        except Exception as e:
            return jsonify({'error': str(e)}), 500

    def call(self):
        self.app.run(host='localhost', port=5000)
    
if __name__ == '__main__':
    instance = server()
    instance.call()