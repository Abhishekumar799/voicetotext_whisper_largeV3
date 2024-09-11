import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import sounddevice as sd
import numpy as np

# Device setup
device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

# Load the model and processor
model_id = "openai/whisper-large-v3"
model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
)
model.to(device)

processor = AutoProcessor.from_pretrained(model_id)

pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    torch_dtype=torch_dtype,
    device=device,
)

# Audio recording parameters
sample_rate = 16000  # Whisper models typically use 16 kHz
duration = 120  # seconds

def record_audio(duration, sample_rate):
    print("Recording...")
    audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype=np.float32)
    sd.wait()  # Wait for the recording to finish
    print("Recording complete.")
    return np.squeeze(audio)

# Main function
def transcribe_live_audio():
    audio = record_audio(duration, sample_rate)  # Record audio from the microphone
    audio_input = {"array": audio, "sampling_rate": sample_rate}

    # Use the model to transcribe
    result = pipe(audio_input)
    print("Transcription:", result["text"])

if __name__ == "__main__":
    transcribe_live_audio()



