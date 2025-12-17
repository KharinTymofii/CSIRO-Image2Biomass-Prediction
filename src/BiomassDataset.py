import os
import cv2
import pandas as pd
import numpy as np

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class BiomassDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        target_cols: list[str],
        img_dir: str,
        transform: transforms.Compose | None = None,
        is_test: bool = False,
        use_log_target: bool = True
    ):
        """
        Args:
            df: DataFrame with image_id, image_path, and target columns
            target_cols: List of target column names
            img_dir: Root directory for images
            transform: torchvision transform pipeline
            is_test: If True, targets are not expected in df
            use_log_target: If True, apply log1p transform to targets
        """
        self.df = df.reset_index(drop=True)
        self.target_cols = target_cols
        self.img_dir = img_dir
        self.transform = transform
        self.is_test = is_test
        self.use_log_target = use_log_target

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        """
        Returns:
            dict with keys:
                - 'left_image': tensor [C, H, W]
                - 'right_image': tensor [C, H, W]
                - 'targets': tensor [n_targets] (if not test)
                - 'image_id': str
        """
        row = self.df.iloc[idx]

        # Load image
        img_path = os.path.join(self.img_dir, row['image_path'].replace(
            'train/', '').replace('test/', ''))
        image = cv2.imread(img_path)

        if image is None:
            raise FileNotFoundError(f"Cannot load image: {img_path}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Split into left and right patches
        # Original image shape: [H, W, C] = [1000, 2000, 3]
        h, w, c = image.shape
        mid_w = w // 2

        left_patch = image[:, :mid_w, :]   # [1000, 1000, 3]
        right_patch = image[:, mid_w:, :]  # [1000, 1000, 3]

        # Convert to PIL Image for torchvision transforms
        left_pil = Image.fromarray(left_patch)
        right_pil = Image.fromarray(right_patch)

        # Apply transforms
        if self.transform:
            left_tensor = self.transform(left_pil)
            right_tensor = self.transform(right_pil)
        else:
            left_tensor = transforms.ToTensor()(left_pil)
            right_tensor = transforms.ToTensor()(right_pil)

        # Prepare output
        output = {
            'left_image': left_tensor,
            'right_image': right_tensor,
            'image_id': row['image_path'].split('/')[-1].replace('.jpg', '')
        }

        # Add targets if not test
        if not self.is_test:
            targets = row[self.target_cols].values.astype(np.float32)

            # Apply log transform if enabled
            if self.use_log_target:
                # log1p handles zeros: log(1+0) = 0
                targets = np.log1p(targets)

            output['targets'] = torch.tensor(targets, dtype=torch.float32)

        return output
