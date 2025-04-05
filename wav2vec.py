import torch
import torchaudio
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
import soundfile as sf
import numpy as np

def load_model():
    model_name = "facebook/wav2vec2-base-960h"
    processor = Wav2Vec2Processor.from_pretrained(model_name)
    model = Wav2Vec2ForCTC.from_pretrained(model_name)
    return processor, model

def speech_to_text(audio_path, processor, model):

    speech, sample_rate = sf.read(audio_path)

 
    if not isinstance(speech, np.ndarray):
        speech = np.array(speech, dtype=np.float32)
    else:
        speech = speech.astype(np.float32)
    

    target_sample_rate = 16000
    if sample_rate != target_sample_rate:
        speech = torch.tensor(speech, dtype=torch.float32)  
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=target_sample_rate)
        speech = resampler(speech).numpy()  # Convert back to NumPy after resampling
        sample_rate = target_sample_rate
    
    # Process input
    inputs = processor(speech, sampling_rate=sample_rate, return_tensors="pt", padding=True)
    
    # Perform inference
    with torch.no_grad():
        logits = model(inputs.input_values).logits
    
    # Decode output
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.batch_decode(predicted_ids)[0]
    
    return transcription

if __name__ == "__main__":
    processor, model = load_model()
    audio_path = "output.wav"  
    result = speech_to_text(audio_path, processor, model)
    print("Transcription:", result)
