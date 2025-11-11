import json
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

# Assuming dect.py is in the same directory
from dect import DeepECT

# 1. Modify the Autoencoder structure (two hidden layers, LeakyReLU)
class Autoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, input_dim)
        )

    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return z, x_hat

# 2. Load embeddings from JSONL
def load_embeddings_from_jsonl(path: Path):
    embeddings = []
    original_indices = []
    fallback_idx = 0
    with path.open("r", encoding="utf-8") as infile:
        print('reading data...')
        for line_number, raw_line in enumerate(infile, start=1):
            stripped = raw_line.strip()
            if not stripped:
                continue
            record = json.loads(stripped)
            if "embedding" not in record:
                raise KeyError(f"Missing 'embedding' key in {path} at line {line_number}")
            embeddings.append(record["embedding"])
            original_indices.append(record.get("idx", fallback_idx))
            fallback_idx += 1
            print(f"{line_number} lines read")
    if not embeddings:
        raise ValueError(f"No embeddings found in {path}")
    tensor = torch.tensor(embeddings, dtype=torch.float32)
    if tensor.ndim != 2 or tensor.shape[1] != 768:
        raise ValueError(f"Expected embeddings with shape (N, 768), got {tuple(tensor.shape)}")
    print(f"successfully loaded tensor with shape {tuple(tensor.shape)}")
    return tensor, original_indices

# 3. Data preparation
DATA_PATH = Path("error_types_embeddings.jsonl")
X, original_indices = load_embeddings_from_jsonl(DATA_PATH)

class ClusteringDataset(Dataset):
    def __init__(self, data):
        self.data = data
    def __len__(self):
        return self.data.shape[0]
    def __getitem__(self, idx):
        return self.data[idx]

dataset = ClusteringDataset(X)
dataloader = DataLoader(dataset, batch_size=256, shuffle=True, num_workers=4)  # Added multi-threading

fulldataloader = DataLoader(dataset, batch_size=512, shuffle=False)

# 4. Multi-GPU training setup
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(DEVICE)
INPUT_DIM = X.shape[1]
LATENT_DIM = 256

embedding_model = Autoencoder(input_dim=INPUT_DIM, latent_dim=LATENT_DIM).to(DEVICE)
dect_model = DeepECT(embedding_model=embedding_model, latent_dim=LATENT_DIM, device=DEVICE)

# 5. Learning rate decay setup
lr = 1e-3
optimizer = optim.Adam(dect_model.parameters(), lr=lr)
lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.1)

# 6. Loss function (cosine distance)
def cosine_distance_loss(x1, x2):
    cos_sim = nn.functional.cosine_similarity(x1, x2, dim=-1)
    return 1 - cos_sim.mean()

# 7. Training loop with loss tracking
losses = []
print("Starting training...")
for epoch in range(5000):  # assuming 5000 iterations
    dect_model.train(
        dataloader=dataloader,
        iterations=5000,
        max_leaves=3000,          # Stop growing when the tree has 10 leaves
        lr = lr,
        split_interval=2,     # Check for splits every 200 iterations
        pruning_threshold=0.05, # Prune nodes with weight < 0.05
        split_count_per_growth=3 # Split the 2 best candidate nodes each time
    )
    running_loss = 0.0
    for i, data in enumerate(dataloader, 0):
        inputs = data.to(DEVICE)
        optimizer.zero_grad()
        _, outputs = dect_model.embedding_model(inputs)
        loss = cosine_distance_loss(inputs, outputs)  # use cosine distance loss
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    losses.append(running_loss / len(dataloader))
    lr_scheduler.step()  # apply learning rate decay

    if epoch % 500 == 0:
        print(f"Epoch [{epoch+1}/5000], Loss: {running_loss / len(dataloader):.4f}")

print("Training finished.")

# 8. Plotting the loss curve
plt.plot(losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss Curve')
plt.savefig("training_loss_curve.png")
plt.show()

# 9. Model saving and loading
MODEL_PATH = "dect_model.pth"
print(f"\nSaving model to {MODEL_PATH}...")
dect_model.save_model(MODEL_PATH)

# Reload the model and verify
new_embedding_model = Autoencoder(input_dim=INPUT_DIM, latent_dim=LATENT_DIM).to(DEVICE)
loaded_dect_model = DeepECT(embedding_model=new_embedding_model, latent_dim=LATENT_DIM, device=DEVICE)
loaded_dect_model.load_model(MODEL_PATH)

# 10. Predict and Save Results
assignments = loaded_dect_model.predict(fulldataloader)
PREDICTIONS_PATH = Path("predictions.jsonl")
assignments_cpu = assignments.detach().cpu().tolist()

if len(assignments_cpu) != len(original_indices):
    raise RuntimeError(f"Length mismatch: assignments={len(assignments_cpu)} vs indices={len(original_indices)}")

with PREDICTIONS_PATH.open("w", encoding="utf-8") as outfile:
    for idx, cluster_id in zip(original_indices, assignments_cpu):
        json.dump({"idx": idx, "cluster": int(cluster_id)}, outfile)
        outfile.write("\n")
print(f"Predictions saved to {PREDICTIONS_PATH}")
