from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import torch
import librosa
import numpy as np
# Load local model and processor
local_model_path = "./whisper-small-local"
model = AutoModelForSpeechSeq2Seq.from_pretrained(local_model_path)
processor = AutoProcessor.from_pretrained(local_model_path)
# Initialize pipeline correctly
pipe = pipeline(
    # Task is already defined here
    "automatic-speech-recognition",  
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    torch_dtype=torch.float32,
    device="cpu",
    # language="en"  # Keep language if needed
)
# Load and process audio
audio_file = "output.wav"
audio, sr = librosa.load(audio_file, sr=16000)
audio_inputs = {"array": np.array(audio), "sampling_rate": sr}
# Transcribe
result = pipe(audio_inputs)
print("Transcribed Text:", result["text"])

