import numpy as np
import torch
from typing import Optional, Tuple
import math
from Model.model import NN
import torch.nn as nn
from surface import State_Encoding
from tqdm import tqdm
import os

class NNFilterDecoder:
    def __init__(self, circuit, model_path: str, d: int, num_layer: int,
                 batch_size: int = 2048, device: str = None, cal_shots_cap: Optional[int] = None):
        self.circuit = circuit
        self.d = d
        self.num_layer = num_layer
        self.batch_size = batch_size
        self.device = torch.device(device if device else ("cuda:0" if torch.cuda.is_available() else "cpu"))
        self.cal_shots_cap = int(cal_shots_cap) if cal_shots_cap else None

        self.model = NN(d=self.d)
        self.model_path = model_path
        state_dict = torch.load(model_path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

        self.coordinates = self._get_detector_coordinates(circuit)
        self.coordinate_array = np.array([self.coordinates[k] for k in range(len(self.coordinates))], dtype=np.int32)
        self.N = 1 + self.num_layer + 1

    # ---------- tensor prep ----------
    def _get_detector_coordinates(self, circuit):
        coords = circuit.get_detector_coordinates()
        d = self.d
        return {k: ((2 * d - int(y)) // 2, int(x) // 2, int(z)) for k, (x, y, z) in coords.items()}

    def _fill_event_tensor(self, syndromes: np.ndarray) -> torch.Tensor:
        B, n_det = syndromes.shape
        E = torch.zeros(B, self.N, self.d + 1, self.d + 1, dtype=torch.float32)
        shot_ids, det_idx = np.nonzero(syndromes)
        coords = self.coordinate_array[det_idx]
        x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]
        E[shot_ids, z, x, y] = 1.0
        return E

    # ---------- inference ----------
    @torch.no_grad()
    def _predict_soft(self, event_tensor_cpu: torch.Tensor) -> np.ndarray:
        results = []
        for i in range(0, event_tensor_cpu.shape[0], self.batch_size):
            batch = event_tensor_cpu[i:i+self.batch_size].to(self.device, non_blocking=True)
            prob = self.model(batch)
            results.append(prob.detach().cpu())
        return torch.cat(results, dim=0).reshape(-1).numpy()

    def _predict_hard_from_soft(self, soft_1d: np.ndarray) -> np.ndarray:
        return (soft_1d > 0.5).astype(np.int32)

    # ---------- public API ----------
    def score_batch(self, syndromes: np.ndarray) -> np.ndarray:
        """Return logit-based confidence score for filtering."""
        if self.cal_shots_cap and len(syndromes) > self.cal_shots_cap:
            idx = np.random.default_rng(42).choice(len(syndromes), size=self.cal_shots_cap, replace=False)
            syndromes = syndromes[idx]

        E = self._fill_event_tensor(syndromes)
        p = self._predict_soft(E)  # probability of logical error
        eps = 1e-9
        conf = abs(np.log((1.0 - np.clip(p, eps, 1 - eps)) / np.clip(p, eps, 1 - eps)))
        return conf

    def accept_mask(self, scores: np.ndarray, tau: float) -> np.ndarray:
        scores = np.asarray(scores).reshape(-1)
        return scores >= float(tau)

    def evaluate_batch(self, syndromes: np.ndarray, actual_observables: np.ndarray, tau: float) -> Tuple[float, float]:
        E = self._fill_event_tensor(syndromes)
        p = self._predict_soft(E)
        preds = self._predict_hard_from_soft(p)
        obs = np.asarray(actual_observables).reshape(-1)

        eps = 1e-9
        conf = np.abs(np.log((1.0 - np.clip(p, eps, 1 - eps)) / np.clip(p, eps, 1 - eps)))
        mask = conf >= float(tau)

        m = int(mask.sum())
        if m == 0:
            return 0.0, 1.0

        cond_logical_err = np.mean((preds[mask] ^ obs[mask]).astype(np.bool_))
        acc_rate = mask.mean()
        return float(acc_rate), float(cond_logical_err)

    @staticmethod
    def quantile_threshold(scores: np.ndarray, target_accept_p: float) -> float:
        """Same quantile logic, now operating in logit score space."""
        s = np.sort(scores.astype(np.float64))
        N = len(s)
        k = int(np.floor((1 - target_accept_p) * N))  # accept high scores
        k = max(0, min(k, N - 1))
        return float(s[k])
    
    # finetune for specific distance and noise parameters
    def finetune(self, p_local: float, p_trans: float,
                 shots: int = 10000,
                 epochs: int = 1,
                 lr: float = 1e-4,
                 finetune_batch_size: int = 256):
        """
        Fine-tune the current decoder model for specific noise parameters (p_local, p_trans).
        Generates new training data, runs several epochs of training, and saves the finetuned model.
        """
        device = self.device
        d = self.d

        # 1. generate data for finetuning
        circuit = State_Encoding(d, self.num_layer, p_local, p_trans)
        sampler = circuit.compile_detector_sampler()
        detector_matrix, actual_observables = sampler.sample(shots=shots, separate_observables=True)

        # 2. transform to event tensor
        E = self._fill_event_tensor(detector_matrix)
        y = torch.tensor(actual_observables, dtype=torch.float32)   # shape (B, 1)

        dataset = torch.utils.data.TensorDataset(E, y)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=finetune_batch_size, shuffle=True)

        # 3. define the optimizer and loss function
        self.model.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.BCELoss()

        # 4. train (finetune) loop
        print(f"[Finetune] Start fine-tuning for d={d}, p_local={p_local}, p_trans={p_trans}")
        total_steps = 0
        total_seen_samples = 0
        for epoch in range(epochs):
            total_loss = 0.0
            pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}", ncols=120)
            for X_batch, y_batch in pbar:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)

                optimizer.zero_grad()
                y_pred = self.model(X_batch)
                loss = criterion(y_pred, y_batch)
                loss.backward()
                optimizer.step()
                batch_size_now = X_batch.shape[0]
                total_seen_samples += batch_size_now
                total_loss += loss.item() * len(X_batch)
                total_steps += 1
                if total_steps % 10 == 0:
                    avg_loss_so_far = total_loss / total_seen_samples
                    pbar.set_postfix_str(f"fine-tuning step={total_steps} avg loss = {avg_loss_so_far:.4f}")
        self.model.eval()

        save_dir = os.path.join(os.path.dirname(self.model_path))
        save_path = os.path.join(save_dir, f"model_{d}_{p_local}_{p_trans}.pt")
        torch.save(self.model.state_dict(), save_path)
        print(f"[Finetune] Model saved to {save_path}")

