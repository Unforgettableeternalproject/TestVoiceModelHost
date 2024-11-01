import os
import edge_tts
from fairseq import checkpoint_utils
import torch
import asyncio
import librosa
import time
from tts.lib.infer_pack.models import (
    SynthesizerTrnMs256NSFsid,
    SynthesizerTrnMs256NSFsid_nono,
    SynthesizerTrnMs768NSFsid,
    SynthesizerTrnMs768NSFsid_nono,
)
from tts.rmvpe import RMVPE
from tts.vc_infer_pipeline import VC
from tts.config import Config

# Initialize configuration
config = Config()
model_root = "tts/weights"
edge_output_filename = os.path.join(os.getcwd(), "tts", "edge_output.mp3")

# Load RVC Model
def model_data(model_name):
    """Loads the specified model and returns its configurations."""
    # global n_spk, tgt_sr, net_g, vc, cpt, version, index_file
    pth_files = [
        os.path.join(model_root, model_name, f)
        for f in os.listdir(os.path.join(model_root, model_name))
        if f.endswith(".pth")
    ]
    if len(pth_files) == 0:
        raise ValueError(f"No pth file found in {model_root}/{model_name}")
    pth_path = pth_files[0]
    print(f"Loading {pth_path}")
    cpt = torch.load(pth_path, map_location="cpu")
    tgt_sr = cpt["config"][-1]
    cpt["config"][-3] = cpt["weight"]["emb_g.weight"].shape[0]  # n_spk
    if_f0 = cpt.get("f0", 1)
    version = cpt.get("version", "v1")
    if version == "v1":
        if if_f0 == 1:
            net_g = SynthesizerTrnMs256NSFsid(*cpt["config"], is_half=config.is_half)
        else:
            net_g = SynthesizerTrnMs256NSFsid_nono(*cpt["config"])
    elif version == "v2":
        if if_f0 == 1:
            net_g = SynthesizerTrnMs768NSFsid(*cpt["config"], is_half=config.is_half)
        else:
            net_g = SynthesizerTrnMs768NSFsid_nono(*cpt["config"])
    else:
        raise ValueError("Unknown version")
    del net_g.enc_q
    net_g.load_state_dict(cpt["weight"], strict=False)
    print("Model loaded")
    net_g.eval().to(config.device)
    if config.is_half:
        net_g = net_g.half()
    else:
        net_g = net_g.float()
    vc = VC(tgt_sr, config)
    # n_spk = cpt["config"][-3]

    index_files = [
        os.path.join(model_root, model_name, f)
        for f in os.listdir(os.path.join(model_root, model_name))
        if f.endswith(".index")
    ]
    if len(index_files) == 0:
        print("No index file found")
        index_file = ""
    else:
        index_file = index_files[0]
        print(f"Index file found: {index_file}")

    return tgt_sr, net_g, vc, version, index_file, if_f0

# Load Hubert model
def load_hubert():
    global hubert_model
    models, _, _ = checkpoint_utils.load_model_ensemble_and_task(["tts/hubert_base.pt"])
    hubert_model = models[0].to(config.device)
    return hubert_model.half() if config.is_half else hubert_model.float()

hubert_model = load_hubert()
rmvpe_model = RMVPE("tts/rmvpe.pt", config.is_half, config.device)

# Text-to-Speech function
def tts(model_name, tts_text, f0_up_key=0, f0_method="rmvpe", index_rate=0, protect=0.33):
    """Processes text input and returns generated TTS audio."""
    print(f"Starting TTS with model: {model_name}")
    tgt_sr, net_g, vc, version, index_file, if_f0 = model_data(model_name)
    print(if_f0)
    t0 = time.time()
    print("Model loaded successfully")

    # Test audio file creation with edge_tts
    print("Generating audio with edge_tts")
    try:
        asyncio.run(
            edge_tts.Communicate(
                tts_text, "en-US-AvaNeural"
            ).save(edge_output_filename)
        )
    except Exception as e:
        print("Error with edge_tts:", e)
        return None, None, None

    try:
        t1 = time.time()
        edge_time = t1 - t0
        audio, sr = librosa.load(edge_output_filename, sr=16000, mono=True)
        duration = len(audio) / sr
        print(f"Audio duration: {duration}s")
        print("Audio loaded successfully")
    except Exception as e:
        print("Error loading audio:", e)
        return None, None, None
    
    f0_up_key = int(f0_up_key)

    if not hubert_model:
        load_hubert()
    if f0_method == "rmvpe":
        vc.model_rmvpe = rmvpe_model

    # Additional processing and voice conversion
    try:
        times = [0, 0, 0]
        audio_opt = vc.pipeline(
            hubert_model,
            net_g,
            0,
            audio,
            edge_output_filename,
            times,
            f0_up_key,
            f0_method,
            index_file,
            index_rate,
            if_f0,
            3,  # Filter radius
            tgt_sr,
            0,  # resample_sr
            0.25,  # rms_mix_rate
            version,
            protect,
            None,
        )
        info = f"Success. Time: edge-tts: {edge_time}s, npy: {times[0]}s, f0: {times[1]}s, infer: {times[2]}s"
        print("TTS pipeline completed successfully")
        return info, edge_output_filename, (tgt_sr, audio_opt)
    except Exception as e:
        print("Error during TTS pipeline:", e)
        return None, None, None
