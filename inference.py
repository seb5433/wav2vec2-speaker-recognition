import torch
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
from torch import nn
import soundfile as sf

def load_model(model_path, processor_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = Wav2Vec2Processor.from_pretrained(processor_path)
    
    # Load model from checkpoint
    model = Wav2Vec2ForCTC.from_pretrained(model_path, ignore_mismatched_sizes=True)

    num_labels = 3  # Number of distinct labels
    model.lm_head = nn.Linear(model.config.hidden_size, num_labels)

    model.to(device)
    model.eval()
    return model, processor, device

def predict(audio_path, model, processor, device):
    # Load audio
    speech, sampling_rate = sf.read(audio_path)
    # Check and resample audio if necessary
    if sampling_rate != 16000:
        raise ValueError(f"Expected 16000Hz audio, but got {sampling_rate}Hz. Please resample.")

    # Process audio
    inputs = processor(speech, return_tensors="pt", padding=True, sampling_rate=16000).input_values.to(device)

    # Model inference
    with torch.no_grad():
        logits = model(inputs).logits

    mean_logits = torch.mean(logits, dim=1)  # Average the logits across all frames
    return mean_logits

model_path = '/home/sebas/Escritorio/work-hugo/wav2vec2-speaker-recognition/epoch'
processor_path = '/home/sebas/Escritorio/work-hugo/wav2vec2-speaker-recognition/epoch'

# Load the model and processor
model, processor, device = load_model(model_path, processor_path)

# Path to the audio file for inference
audio_path = 'nino.wav'

# Perform prediction
predicted_ids = predict(audio_path, model, processor, device)
print(f"Predicted class IDs: {predicted_ids}")
