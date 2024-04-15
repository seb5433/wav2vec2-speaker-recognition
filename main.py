import torch
from torch import nn
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from torch.utils.data import DataLoader, random_split
import os
from src import AudioDataset

# Define the device to run the model on
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Cargar el procesador y el modelo pre-entrenado
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base")
model.to(device)

# Añadir una capa de clasificación en la parte superior de wav2vec 2.0
# Supongamos que tienes 3 clases distintas: TEACHER, CHILD, OTHER
num_labels = 3  # Número de hablantes a clasificar
model.lm_head = nn.Linear(model.config.hidden_size, num_labels)  # Reemplazar la cabeza de LM por una de clasificación
model.lm_head.to(device)

# Load the dataset
label_dict = {"TEACHER": 0, "CHILD": 1, "OTHER": 2}
full_dataset = AudioDataset(csv_path='./data/metadata.csv', label_dict=label_dict)

# Split the dataset into train and test sets
batch_size = 32  # Define the batch size
train_size = int(len(full_dataset) * 0.8)  # 80% of the data for training
test_size = len(full_dataset) - train_size  # Remaining 20% for testing
train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])

# Create DataLoaders for both train and test sets
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=full_dataset.collate_fn)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=full_dataset.collate_fn)

# Definir una función de pérdida y un optimizador
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

def train(model, train_loader, processor, loss_fn, optimizer, device):
    model.train()
    for batch in train_loader:
        speech, labels = batch
        speech = speech.to(device)  # Ensure speech data is on the correct device
        labels = labels.to(device)  # Ensure labels are on the correct device

        print (f"Speech Shape: {speech.shape}") 

        # Process the input batch through the processor
        inputs = processor(speech, return_tensors="pt", padding=True, sampling_rate=16000)
        input_values = inputs.input_values.to(device)  # Get the processed inputs ready for the model

        # Debug: Check input dimension
        print(f"Input Values Shape: {input_values.shape}")  # Should be [batch_size, seq_length]

        # Reshape to remove any extra singleton dimension
        if input_values.dim() == 3 and input_values.size(0) == 1:
            input_values = input_values.squeeze(0)

        print(f"Processed Input Values Shape: {input_values.shape}")

        # Forward pass
        outputs = model(input_values)
        logits = outputs.logits

        print (f"Logits Shape: {logits.shape}")
        print (f"Labels Shape: {labels.shape}")

        # Compute loss
        loss = loss_fn(logits, labels)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"Loss: {loss.item()}")

# Entrenar el modelo
train(model, train_loader, processor, loss_fn, optimizer, device)

# Guardar el modelo fine-tuneado
save_directory = "./"
os.makedirs(save_directory, exist_ok=True)
model.save_pretrained(save_directory)
processor.save_pretrained(save_directory)
