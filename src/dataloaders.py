import pandas as pd
from torch.utils.data import Dataset, DataLoader, Subset


def get_fold_loaders(
    df: pd.DataFrame,
    dataset: Dataset,
    fold: int,
    batch_size: int,
    num_workers: int = 0,
    shuffle_train: bool = True,
) -> tuple[DataLoader, DataLoader]:
    """
    Create train and validation DataLoaders for a specific fold.

    Args:
        df: DataFrame with 'fold' column (result of create_folds_with_date_groups)
        dataset: PyTorch Dataset instance with all samples
        fold: Fold index to use for validation (0 to n_folds-1)
        batch_size: Batch size for both loaders
        num_workers: Number of workers for data loading (0 for single process)
        shuffle_train: Whether to shuffle training data
        pin_memory: Whether to pin memory for GPU transfer

    Returns:
        tuple[DataLoader, DataLoader]: (train_loader, val_loader)

    Example:
        >>> df_with_folds = create_folds_with_date_groups(df_pivoted, n_folds=5)
        >>> train_dataset = BiomassDataset(df_pivoted, target_cols, img_dir, transform)
        >>> train_loader, val_loader = get_fold_loaders(
        ...     df=df_with_folds,
        ...     dataset=train_dataset,
        ...     fold=0,
        ...     batch_size=4,
        ...     num_workers=4
        ... )
    """
    # Get indices for train and validation
    train_indices = df[df['fold'] != fold].index.tolist()
    val_indices = df[df['fold'] == fold].index.tolist()

    # Create subsets
    train_subset = Subset(dataset, train_indices)
    val_subset = Subset(dataset, val_indices)

    # Create DataLoaders
    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=shuffle_train,
        num_workers=num_workers,
    )

    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,  # Never shuffle validation
        num_workers=num_workers,
    )

    return train_loader, val_loader


def print_fold_loader_info(
    df: pd.DataFrame,
    fold: int,
    train_loader: DataLoader,
    val_loader: DataLoader
) -> None:
    """
    Print information about train/val loaders for debugging.

    Args:
        df: DataFrame with fold information
        fold: Fold index
        train_loader: Training DataLoader
        val_loader: Validation DataLoader
    """
    train_samples = len(df[df['fold'] != fold])
    val_samples = len(df[df['fold'] == fold])

    print(f"\nFold {fold} DataLoaders Info:")
    print(f"  Train samples: {train_samples}")
    print(f"  Val samples: {val_samples}")
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")
    print(f"  Train batch size: {train_loader.batch_size}")
    print(f"  Val batch size: {val_loader.batch_size}")
