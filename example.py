import json
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from pathlib import Path
from torch.cuda import amp
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

# Assuming dect.py is in the same directory
from dect import DeepECT


class ResidualBlock(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.linear1 = nn.Linear(dim, dim)
        self.activation = nn.GELU()
        self.linear2 = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.linear2(self.activation(self.linear1(x)))
        return self.norm(x + residual)


class Autoencoder(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int, noise_std: float = 0.05) -> None:
        super().__init__()
        self.noise_std = noise_std

        self.input_norm = nn.LayerNorm(input_dim)

        self.enc_fc1 = nn.Linear(input_dim, 512)
        self.enc_act1 = nn.GELU()
        self.enc_dropout1 = nn.Dropout(0.1)
        self.enc_res1 = ResidualBlock(512)

        self.enc_fc2 = nn.Linear(512, 256)
        self.enc_act2 = nn.GELU()
        self.enc_dropout2 = nn.Dropout(0.1)
        self.enc_res2 = ResidualBlock(256)

        self.enc_latent = nn.Linear(256, latent_dim)

        self.dec_fc1 = nn.Linear(latent_dim, 256)
        self.dec_act1 = nn.GELU()
        self.dec_res1 = ResidualBlock(256)
        self.dec_dropout1 = nn.Dropout(0.1)

        self.dec_fc2 = nn.Linear(256, 512)
        self.dec_act2 = nn.GELU()
        self.dec_res2 = ResidualBlock(512)

        self.dec_out = nn.Linear(512, input_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.training and self.noise_std > 0:
            noise = torch.randn_like(x) * self.noise_std
            x_noisy = x + noise
        else:
            x_noisy = x

        h = self.input_norm(x_noisy)
        h = self.enc_dropout1(self.enc_act1(self.enc_fc1(h)))
        h = self.enc_res1(h)
        h = self.enc_dropout2(self.enc_act2(self.enc_fc2(h)))
        h = self.enc_res2(h)
        z = self.enc_latent(h)
        z = F.normalize(z, p=2, dim=-1)

        h = self.dec_act1(self.dec_fc1(z))
        h = self.dec_res1(h)
        h = self.dec_dropout1(h)
        h = self.dec_act2(self.dec_fc2(h))
        h = self.dec_res2(h)
        x_hat = self.dec_out(h)
        return z, x_hat

# 2. Load embeddings from JSONL
def load_embeddings_from_jsonl(path: Path):
    embeddings = []
    original_indices = []
    phrases = []
    truncated_embeddings = []
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
            if "phrase" not in record:
                raise KeyError(f"Missing 'phrase' key in {path} at line {line_number}")
            embedding_values = record["embedding"]
            if not isinstance(embedding_values, list):
                raise TypeError(f"Expected 'embedding' to be a list at line {line_number}")
            if len(embedding_values) < 768:
                raise ValueError(
                    f"Expected embedding length >= 768 (found {len(embedding_values)}) at line {line_number}"
                )
            truncated = embedding_values[:768]
            truncated_embeddings.append(truncated)
            embeddings.append(truncated)
            original_indices.append(record.get("idx", fallback_idx))
            phrases.append(record["phrase"])
            fallback_idx += 1
            print(f"{line_number} lines read")
    if not embeddings:
        raise ValueError(f"No embeddings found in {path}")
    tensor = torch.tensor(embeddings, dtype=torch.float32)
    if tensor.ndim != 2 or tensor.shape[1] != 768:
        raise ValueError(f"Expected embeddings with shape (N, 768), got {tuple(tensor.shape)}")
    print(f"successfully loaded tensor with shape {tuple(tensor.shape)}")
    return tensor, original_indices, phrases, truncated_embeddings

# 3. Data preparation
DATA_PATH = Path("error_types_embeddings.jsonl")
X, original_indices, phrases, truncated_embeddings = load_embeddings_from_jsonl(DATA_PATH)

class ClusteringDataset(Dataset):
    def __init__(self, data):
        self.data = data
    def __len__(self):
        return self.data.shape[0]
    def __getitem__(self, idx):
        return self.data[idx]

dataset = ClusteringDataset(X)
pretrain_size = max(1, int(len(dataset) * 0.4))
finetune_size = max(0, len(dataset) - pretrain_size)
generator = torch.Generator().manual_seed(42)
pretrain_dataset, _ = random_split(dataset, [pretrain_size, finetune_size], generator=generator)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pin_memory = DEVICE.type == "cuda"
print(DEVICE)
INPUT_DIM = X.shape[1]
LATENT_DIM = 128

micro_batch_size = min(1024, max(1, len(dataset)))
pretrain_loader = DataLoader(pretrain_dataset, batch_size=micro_batch_size, shuffle=True, num_workers=4, pin_memory=pin_memory)
dataloader = DataLoader(dataset, batch_size=micro_batch_size, shuffle=True, num_workers=4, pin_memory=pin_memory)
fulldataloader = DataLoader(dataset, batch_size=2048, shuffle=False, num_workers=4, pin_memory=pin_memory)

embedding_model = Autoencoder(input_dim=INPUT_DIM, latent_dim=LATENT_DIM).to(DEVICE)

lr = 1e-3


def cosine_distance_loss(x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
    cos_sim = F.cosine_similarity(x1, x2, dim=-1)
    return 1 - cos_sim.mean()


def build_cosine_with_warmup_scheduler(optimizer: optim.Optimizer, total_steps: int, warmup_steps: int = 5000, last_step: int = -1):
    warmup_steps = min(warmup_steps, total_steps)

    def lr_lambda(step: int) -> float:
        step_index = step + 1
        if total_steps <= 0:
            return 1.0
        if step_index <= warmup_steps and warmup_steps > 0:
            return float(step_index) / float(max(1, warmup_steps))
        if total_steps == warmup_steps:
            return 1.0
        progress = (step_index - warmup_steps) / float(max(1, total_steps - warmup_steps))
        progress = min(max(progress, 0.0), 1.0)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda, last_epoch=last_step)


# Autoencoder pretraining on 40% of the data
use_amp = DEVICE.type == "cuda"
scaler = amp.GradScaler(enabled=use_amp)
pretrain_optimizer = optim.AdamW(
    embedding_model.parameters(),
    lr=lr,
    betas=(0.9, 0.95),
    weight_decay=0.01,
)
pretrain_optimizer.zero_grad(set_to_none=True)

accumulation_steps = None
total_pretrain_steps = None
pretrain_scheduler = None
pretrain_epochs = 3
pretrain_losses = []

print("Starting autoencoder pretraining...")
for epoch in range(pretrain_epochs):
    embedding_model.train()
    running_loss = 0.0
    completed_steps = 0
    for batch_idx, batch in enumerate(pretrain_loader):
        inputs = batch.to(DEVICE, non_blocking=pin_memory)

        if accumulation_steps is None:
            desired_global_batch = 4096 if use_amp else max(inputs.size(0), 1024)
            accumulation_steps = max(1, desired_global_batch // inputs.size(0))
            total_pretrain_steps = math.ceil(len(pretrain_loader) / accumulation_steps) * pretrain_epochs
            pretrain_scheduler = build_cosine_with_warmup_scheduler(
                pretrain_optimizer,
                total_steps=total_pretrain_steps,
                warmup_steps=5000,
            )

        with amp.autocast(enabled=use_amp):
            _, outputs = embedding_model(inputs)
            loss = cosine_distance_loss(inputs, outputs)

        scaled_loss = loss / accumulation_steps
        scaler.scale(scaled_loss).backward()

        should_step = ((batch_idx + 1) % accumulation_steps == 0) or (batch_idx + 1 == len(pretrain_loader))
        if should_step:
            scaler.unscale_(pretrain_optimizer)
            torch.nn.utils.clip_grad_norm_(embedding_model.parameters(), 1.0)
            scaler.step(pretrain_optimizer)
            scaler.update()
            pretrain_optimizer.zero_grad(set_to_none=True)
            if pretrain_scheduler is not None:
                pretrain_scheduler.step()
            completed_steps += 1

        running_loss += loss.item()

    epoch_loss = running_loss / max(1, completed_steps)
    pretrain_losses.append(epoch_loss)
    print(f"Pretrain Epoch [{epoch + 1}/{pretrain_epochs}], Loss: {epoch_loss:.4f}")

print("Autoencoder pretraining finished.")

# Initialize DeepECT with pretrained embeddings
embedding_model.eval()
latent_vectors = []
with torch.no_grad():
    for batch in fulldataloader:
        inputs = batch.to(DEVICE, non_blocking=pin_memory)
        z, _ = embedding_model(inputs)
        latent_vectors.append(z.detach().cpu())

latent_vectors = torch.cat(latent_vectors, dim=0)
dect_model = DeepECT(
    embedding_model=embedding_model,
    latent_dim=LATENT_DIM,
    embedding_model_loss=cosine_distance_loss,
    device=DEVICE,
)
dect_model.initialize_tree_from_embeddings(latent_vectors)
embedding_model.train()

print("Starting joint DeepECT training...")
joint_loss_history, leaf_purity_history = dect_model.train(
    dataloader=dataloader,
    iterations=5000,
    lr=lr,
    max_leaves=3000,
    split_interval=2,
    pruning_threshold=0.05,
    split_count_per_growth=3,
    evaluation_loader=fulldataloader,
)
print("Joint training finished.")

# Plotting the loss curve
plt.figure()
plt.plot(range(1, len(pretrain_losses) + 1), pretrain_losses, label="Pretraining (epoch avg)")
if joint_loss_history:
    offset = len(pretrain_losses)
    joint_steps = [offset + step + 1 for step in range(len(joint_loss_history))]
    plt.plot(joint_steps, joint_loss_history, label="Joint training (per step)")
plt.xlabel('Step')
plt.ylabel('Loss')
plt.title('Training Loss Curve')
plt.legend()
plt.savefig("training_loss_curve.png")
plt.show()

# Plotting the leaf purity curve
if leaf_purity_history:
    plt.figure()
    epochs = range(1, len(leaf_purity_history) + 1)
    plt.plot(epochs, leaf_purity_history, marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Average Leaf Purity')
    plt.title('Leaf Purity per Epoch')
    plt.grid(True)
    plt.savefig("leaf_purity_curve.png")
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
PRED_TEXT_PATH = Path("pred_text.jsonl")
PRED_EMB_PATH = Path("pred_emb.jsonl")
assignments_cpu = assignments.detach().cpu().tolist()

if len(assignments_cpu) != len(original_indices):
    raise RuntimeError(f"Length mismatch: assignments={len(assignments_cpu)} vs indices={len(original_indices)}")

with PRED_TEXT_PATH.open("w", encoding="utf-8") as text_file, PRED_EMB_PATH.open("w", encoding="utf-8") as emb_file:
    for idx, phrase, embedding_list, cluster_id in zip(original_indices, phrases, truncated_embeddings, assignments_cpu):
        json.dump({"idx": idx, "phrase": phrase, "cluster": int(cluster_id)}, text_file)
        text_file.write("\n")
        json.dump({"idx": idx, "embedding": embedding_list, "cluster": int(cluster_id)}, emb_file)
        emb_file.write("\n")
print(f"Predictions saved to {PRED_TEXT_PATH} and {PRED_EMB_PATH}")
