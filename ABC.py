import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import argparse
from typing import Dict, List, Tuple

from model.swin_unet import SwinTransformerSys
from utils import torch_ps

class CMBDataset(Dataset):
    """Dataset class for CMB map patches."""
    
    def __init__(self, file_ids: List[str], base_dir: str):
        """
        Initialize the dataset.
        
        Args:
            file_ids: List of file identifiers
            base_dir: Base directory containing the data files
        """
        self.file_ids = file_ids
        self.base_dir = base_dir
        self.patches_per_map = 48

    def __len__(self) -> int:
        return len(self.file_ids) * self.patches_per_map

    def __getitem__(self, index: int) -> torch.Tensor:
        """Get power spectrum for a specific patch."""
        file_idx = self.file_ids[index // self.patches_per_map]
        patch_idx = index % self.patches_per_map
        
        data = torch.load(os.path.join(self.base_dir, "statistics", file_idx))
        return data[patch_idx]

class ABCInference:
    """Class for performing ABC inference on CMB data."""
    
    def __init__(self, 
                 model_path: str,
                 device: str = 'cpu',
                 patches_per_map: int = 48,
                 samples_per_map: int = 7):
        """
        Initialize ABC inference.
        
        Args:
            model_path: Path to pretrained model weights
            device: Computing device ('cpu' or 'cuda')
            patches_per_map: Number of patches per map
            samples_per_map: Number of samples per map
        """
        self.device = device
        self.patches_per_map = patches_per_map
        self.total_samples = patches_per_map * samples_per_map
        
        # Initialize model
        self.model = self._load_model(model_path)
        
    def _load_model(self, model_path: str) -> torch.nn.Module:
        """Load pretrained model."""
        checkpoint = torch.load(model_path)
        model = SwinTransformerSys(
            skip_connect=True,
            drop_rate=0.1,
            attn_drop_rate=0.1,
            Fourier=False,
            last_bias=True,
            VAE=True,
            latent_dim=100
        ).to(self.device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        return model

    @staticmethod
    def extract_r_value(filename: str) -> float:
        """Extract r value from filename."""
        start = filename.index("r = ")
        end = filename.index("<start>")
        return float(filename[start + 4 : end])

    def find_nearest_r(self, r_values: torch.Tensor, target_r: torch.Tensor) -> torch.Tensor:
        """Find nearest available r value."""
        return r_values[(torch.abs(r_values.unsqueeze(0) - target_r.unsqueeze(1))).argmin(dim=1)]

    def sample_data_indices(self, 
                          sample_r: torch.Tensor,
                          r_list: List[float],
                          sample_record: Dict[float, List[int]]) -> List[int]:
        """Generate sample indices ensuring no duplicates."""
        sample_idx_list = []
        
        for r_val in sample_r:
            while True:
                sample_idx = torch.randint(0, self.total_samples, (1,)).item()
                r_key = round(r_val.item(), 5)
                
                if sample_idx not in sample_record[r_key]:
                    if len(sample_record[r_key]) >= self.total_samples:
                        raise RuntimeError(f"Exhausted all samples for r={r_key}")
                    
                    sample_record[r_key].append(sample_idx)
                    final_idx = r_list.index(r_val) * self.total_samples + sample_idx
                    sample_idx_list.append(final_idx)
                    break
                    
        return sample_idx_list

    def run_inference(self,
                     observed_data: torch.Tensor,
                     available_files: List[str],
                     base_dir: str,
                     n_samples: int = 40000,
                     threshold: float = 0.05,
                     batch_size: int = 48) -> torch.Tensor:
        """
        Run ABC inference.
        
        Args:
            observed_data: Observed Q/U maps
            available_files: List of available data files
            base_dir: Base directory for data
            n_samples: Number of samples for ABC
            threshold: Acceptance threshold
            batch_size: Batch size for data loading
            
        Returns:
            Accepted r values
        """
        # Get observed power spectrum
        with torch.no_grad():
            B_pred, _, _ = self.model(observed_data.unsqueeze(0))
        observed_ps = torch_ps(B_pred, self.device).squeeze().log()

        # Prepare r values
        r_list = sorted(list(set([
            self.extract_r_value(f) for f in available_files 
            if self.extract_r_value(f) <= 0.3
        ])))
        r_tensor = torch.tensor(r_list, device=self.device)
        sample_record = {round(r, 5): [] for r in r_list}

        # Sample from prior
        sample_r = torch.empty(n_samples, device=self.device).uniform_(0.001, 0.301)
        nearest_r = self.find_nearest_r(r_tensor, sample_r)
        
        # Get sample indices
        indices = self.sample_data_indices(nearest_r, r_list, sample_record)
        
        # Prepare dataset
        dataset = CMBDataset(available_files, base_dir)
        subset = torch.utils.data.Subset(dataset, indices)
        dataloader = DataLoader(subset, batch_size=batch_size, shuffle=False, num_workers=8)

        # Collect power spectra
        all_ps = []
        for data in tqdm(dataloader, desc="Processing samples"):
            all_ps.append(data)
        all_ps = torch.cat(all_ps, dim=0)

        # Calculate errors and find accepted samples
        errors = (all_ps - observed_ps.view(1, -1)).abs().sum(dim=1)
        threshold_value = torch.quantile(errors, threshold)
        accept_mask = errors < threshold_value
        
        return nearest_r[accept_mask]

def plot_posterior(accept_r: torch.Tensor, 
                  true_r: float, 
                  patch_id: int, 
                  threshold: float,
                  output_dir: str):
    """Plot and save posterior distribution."""
    fig, ax = plt.subplots(figsize=(5, 5))
    sns.histplot(accept_r.cpu().numpy(), stat='density')
    sns.kdeplot(accept_r.cpu().numpy())
    ax.axvline(x=true_r, linestyle="--", label='True r')
    ax.set_xlabel('r')
    ax.set_ylabel('Density')
    ax.legend()
    
    plt.savefig(
        f'{output_dir}/r={true_r}_distribution_{patch_id}_thr={threshold}.png',
        dpi=200
    )
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='CMB ABC Inference')
    parser.add_argument('--batch_id', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--threshold', type=float, default=0.05)
    parser.add_argument('--true_r', type=float, default=0.1)
    args = parser.parse_args()

    # Initialize inference
    inference = ABCInference(
        model_path='weight/ABCMB_v1.pt',
        device='cpu'
    )

    # Process each batch
    for batch_num in range(48 // args.batch_size):
        piece = (args.batch_id) * (48 // args.batch_size) + batch_num
        
        # Load observed data
        data_file = f'r = {args.true_r:.4f}<start>160_sqdeg_224x224_pix_{51 if piece <= 47 else 52}.pt'
        observed_data = torch.load(f'output/inference_map/all_map/{data_file}')[0:2, piece % 48].to(torch.float32)

        # Get available files for inference
        available_files = [
            f for f in os.listdir('output/inference_map/all_map/')
            if not f.endswith('51.pt') and not f.endswith('52.pt')
        ]
        available_files.sort()

        # Run inference
        accept_r = inference.run_inference(
            observed_data=observed_data,
            available_files=available_files,
            base_dir='output/inference_map/',
            threshold=args.threshold
        )

        # Save results
        plot_posterior(accept_r, args.true_r, piece, args.threshold, 'Inference/figure')
        torch.save(accept_r, f'Inference/sample/accept_r={args.true_r}_{piece}_thr={args.threshold}.pt')

if __name__ == "__main__":
    main()