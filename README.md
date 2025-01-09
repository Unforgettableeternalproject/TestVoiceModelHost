# TestVoiceModelHost

## Overview

This is a project aimed at hosting a custom voice model TTS API on a server. This project allows users to request TTS services using the custom voice model.

## Features

- **TTS Functionality**: Provides text-to-speech services using a custom voice model.
- **API Hosting**: Hosts an API on the server, allowing users to send requests and receive voice output.
- **Access Control**: Only users with an access code can use the TTS service.
- **Error Handling**: Captures and displays common API errors.

## Project Structure

```graphql
│ TestVoiceModelHost/ 
├── backend/ 
│   ├── caller.py  # API caller implementation
│   └── server.py
│ 
├── tts/ 
│   ├── weights/  # Pre-trained model weights
│   ├── lib/
│   ├── config.py
│   ├── app.py
│   ├── rmvpe.py
│   └── vc_infer__pipeline.py
│ 
├── Entry.py  # Entry point 
└── README.md  # Project documentation
```

## Reference

This project is inspired by the repository [rvc-tts-webui](https://github.com/litagin02/rvc-tts-webui).

## Disclaimer

This project is **not** intended for public use. 

The model within this project is not public, and the server IP and port are fake. 

For most part, it is for demonstration purposes only.
