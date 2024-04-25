import torch
from torch import nn
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from torch.utils.data import DataLoader, random_split
import os
from src import AudioDataset

# Training loop
num_epochs = 30

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load processor and pretrained model
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base")
model.to(device)

# Add a classification layer
num_labels = 3  # Number of distinct labels
model.lm_head = nn.Linear(model.config.hidden_size, num_labels)  # Replace LM head with a classification head
model.lm_head.to(device)

# Load dataset
label_dict = {"TEACHER": 0, "CHILD": 1, "OTHER": 2}
full_dataset = AudioDataset(csv_path='./data/metadata.csv', label_dict=label_dict)

# Split dataset into train and test sets
batch_size = 32
train_size = int(0.8 * len(full_dataset))
test_size = len(full_dataset) - train_size
train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])

# Data loaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=full_dataset.collate_fn)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=full_dataset.collate_fn)

# Loss function and optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

def train_epoch(model, loader, processor, loss_fn, optimizer, device):
    model.train()
    total_loss = 0
    for batch in loader:
        speech, labels = batch
        inputs = processor(speech, return_tensors="pt", padding=True, sampling_rate=16000).input_values.to(device)
        labels = labels.to(device)

        # Reshape if the batch size is 1 to remove singleton dimension
        if inputs.dim() == 3 and inputs.size(0) == 1:
            inputs = inputs.squeeze(0)

        outputs = model(inputs)
        loss = loss_fn(outputs.logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(loader)

def evaluate(model, loader, processor, loss_fn, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in loader:
            speech, labels = batch
            inputs = processor(speech, return_tensors="pt", padding=True, sampling_rate=16000).input_values.to(device)
            labels = labels.to(device)

            # Reshape if the batch size is 1 to remove singleton dimension
            if inputs.dim() == 3 and inputs.size(0) == 1:
                inputs = inputs.squeeze(0)

            outputs = model(inputs)
            loss = loss_fn(outputs.logits, labels)
            total_loss += loss.item()

    return total_loss / len(loader)



def train_and_evaluate(model, train_loader, test_loader, processor, loss_fn, optimizer, device, num_epochs=10, save_path="./epoch"):
    best_loss = float('inf')  # Initialize with a very high value
    for epoch in range(num_epochs):
        train_loss = train_epoch(model, train_loader, processor, loss_fn, optimizer, device)
        test_loss = evaluate(model, test_loader, processor, loss_fn, device)
        print(f"Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}")

        # Check if the current model is the best one
        if test_loss < best_loss:
            best_loss = test_loss
            print(f"New best model found at epoch {epoch+1} with test loss {test_loss:.4f}. Saving model...")
            model.save_pretrained(save_path)
            processor.save_pretrained(save_path)

print ("Training model...")
train_and_evaluate(model, train_loader, test_loader, processor, loss_fn, optimizer, device, num_epochs=num_epochs)
