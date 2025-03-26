from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
local_model_path = "./whisper-small-local"
model = AutoModelForSpeechSeq2Seq.from_pretrained("openai/whisper-small")
processor = AutoProcessor.from_pretrained("openai/whisper-small")

# Save the model and processor locally
model.save_pretrained(local_model_path)
processor.save_pretrained(local_model_path)
