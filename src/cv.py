import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold


def get_season(date_str: str) -> str:
    """
    Convert date string to season (Australian seasons).

    Args:
        date_str: Date in format 'YYYY/M/D' or 'YYYY/MM/DD'

    Returns:
        Season name: 'Summer', 'Autumn', 'Winter', 'Spring'
    """
    # Parse month from date string
    month = int(date_str.split('/')[1])

    # Australian seasons (Southern Hemisphere)
    if month in [12, 1, 2]:
        return 'Summer'
    elif month in [3, 4, 5]:
        return 'Autumn'
    elif month in [6, 7, 8]:
        return 'Winter'
    else:  # 9, 10, 11
        return 'Spring'


def create_mass_bins(df: pd.DataFrame, n_bins: int = 5, mass_col: str = 'Dry_Total_g') -> pd.Series:
    """
    Create bins for mass stratification using quantiles.

    Args:
        df: DataFrame with mass column
        n_bins: Number of bins for stratification
        mass_col: Column name for mass values

    Returns:
        Series with bin labels for each sample
    """
    if mass_col not in df.columns:
        raise ValueError(
            f"Column '{mass_col}' not found in DataFrame. Available columns: {df.columns.tolist()}")

    # Create bins using quantiles to ensure equal distribution
    mass_bins = pd.qcut(
        df[mass_col],
        q=n_bins,
        labels=[f'mass_bin_{i}' for i in range(n_bins)],
        duplicates='drop'  # Handle duplicate bin edges
    )

    return mass_bins


def create_stratification_groups(
    df: pd.DataFrame,
    use_mass_stratification: bool = True,
    n_mass_bins: int = 5,
    mass_col: str = 'Dry_Total_g',
    use_species_stratification: bool = False
) -> pd.DataFrame:
    """
    Create stratification groups based on State, optionally Species and Mass.

    Args:
        df: DataFrame with 'Sampling_Date', 'State', 'Species' columns
        use_mass_stratification: Whether to include mass bins in stratification
        n_mass_bins: Number of bins for mass stratification
        mass_col: Column name for mass values
        use_species_stratification: Whether to include Species in stratification

    Returns:
        DataFrame with added 'Season', 'mass_bin', and 'strat_group' columns
    """
    df = df.copy()

    # Add season column for analysis
    df['Season'] = df['Sampling_Date'].apply(get_season)

    # Create mass bins if requested
    if use_mass_stratification:
        df['mass_bin'] = create_mass_bins(
            df, n_bins=n_mass_bins, mass_col=mass_col)
    else:
        df['mass_bin'] = 'no_bin'

    # Create stratification column
    strat_parts = [df['State'].astype(str)]
    
    if use_species_stratification:
        strat_parts.append(df['Species'].astype(str))
    
    if use_mass_stratification:
        strat_parts.append(df['mass_bin'].astype(str))
    
    df['strat_group'] = '_'.join([f'{part}' for part in strat_parts])
    for i in range(1, len(strat_parts)):
        df['strat_group'] = str(df['strat_group']) + '_' + strat_parts[i]

    return df


def create_folds_with_date_groups(
    df: pd.DataFrame,
    n_folds: int = 5,
    random_state: int = 42,
    use_mass_stratification: bool = True,
    n_mass_bins: int = 5,
    mass_col: str = 'Dry_Total_g',
    verbose: bool = True
) -> pd.DataFrame:
    """
    Create folds using StratifiedGroupKFold.
    - Groups by Sampling_Date (prevents date leakage)
    - Stratifies by State + Species + Mass bins (balances distributions)

    Args:
        df: DataFrame with 'Sampling_Date', 'State', 'Species' columns
        n_folds: Number of folds
        random_state: Random seed for reproducibility
        use_mass_stratification: Whether to stratify by mass distribution
        n_mass_bins: Number of bins for mass stratification (5 = quintiles)
        mass_col: Column name for mass values
        verbose: Print fold statistics

    Returns:
        DataFrame with added 'fold' column
    """
    df = create_stratification_groups(
        df,
        use_mass_stratification=use_mass_stratification,
        n_mass_bins=n_mass_bins,
        mass_col=mass_col
    )

    # Use StratifiedGroupKFold
    # groups: Sampling_Date (prevents same date in train/val)
    # stratify: State + Species + Mass_Bin (balances across folds)
    sgkf = StratifiedGroupKFold(
        n_splits=n_folds,
        shuffle=True,
        random_state=random_state
    )

    # Initialize fold column
    df['fold'] = -1

    # Get groups (Sampling_Date) and stratification labels (State_Species_MassBin)
    groups = df['Sampling_Date'].to_numpy()
    strat_labels = df['strat_group'].to_numpy()

    # Split data
    for fold_idx, (train_idx, val_idx) in enumerate(
        sgkf.split(df, y=strat_labels, groups=groups)
    ):
        df.loc[df.index[val_idx], 'fold'] = fold_idx

        if verbose:
            print(f"\nFold {fold_idx}:")
            print(f"  Train samples: {len(train_idx)}")
            print(f"  Val samples: {len(val_idx)}")

            # Check unique dates don't overlap
            train_dates = set(df.iloc[train_idx]['Sampling_Date'].unique())
            val_dates = set(df.iloc[val_idx]['Sampling_Date'].unique())
            overlap = train_dates & val_dates

            if overlap:
                print(f"  WARNING: {len(overlap)} dates overlap!")
            else:
                print(
                    f"  ✓ No date overlap (train: {len(train_dates)} dates, val: {len(val_dates)} dates)")

    if verbose:
        verify_fold_quality(df, n_folds, mass_col=mass_col)

    return df


def verify_fold_quality(df: pd.DataFrame, n_folds: int, mass_col: str = 'Dry_Total_g') -> None:
    """
    Verify that folds have balanced distributions.

    Args:
        df: DataFrame with 'fold', 'Season', 'State', 'Species' columns
        n_folds: Number of folds
        mass_col: Column name for mass values
    """
    print("FOLD QUALITY VERIFICATION")

    for fold in range(n_folds):
        fold_df = df[df['fold'] == fold]

        print(f"\nFold {fold}:")
        print(f"  Samples: {len(fold_df)}")

        # State distribution
        print(f"  State distribution:")
        state_dist = fold_df['State'].value_counts(normalize=True).round(3)
        for state, prop in state_dist.items():
            print(f"    {state}: {prop:.1%}")

        # Species distribution
        print(f"  Species distribution:")
        species_dist = fold_df['Species'].value_counts(normalize=True).round(3)
        for species, prop in species_dist.items():
            print(f"    {species}: {prop:.1%}")

        # Season distribution
        print(f"  Season distribution:")
        season_dist = fold_df['Season'].value_counts(normalize=True).round(3)
        for season, prop in season_dist.items():
            print(f"    {season}: {prop:.1%}")

        # Mass bin distribution
        if 'mass_bin' in fold_df.columns and fold_df['mass_bin'].iloc[0] != 'no_bin':
            print(f"  Mass bin distribution:")
            mass_bin_dist = fold_df['mass_bin'].value_counts(
                normalize=True).sort_index().round(3)
            for bin_name, prop in mass_bin_dist.items():
                print(f"    {bin_name}: {prop:.1%}")

        # Mass statistics
        if mass_col in fold_df.columns:
            print(f"  {mass_col} statistics:")
            print(f"    Mean: {fold_df[mass_col].mean():.2f}")
            print(f"    Median: {fold_df[mass_col].median():.2f}")
            print(f"    Std: {fold_df[mass_col].std():.2f}")
            print(f"    Min: {fold_df[mass_col].min():.2f}")
            print(f"    Max: {fold_df[mass_col].max():.2f}")

        # Unique dates
        unique_dates = fold_df['Sampling_Date'].nunique()
        print(f"  Unique dates: {unique_dates}")


def get_fold_data(df: pd.DataFrame, fold: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Get train/val split for specific fold.

    Args:
        df: DataFrame with 'fold' column
        fold: Fold index to use as validation

    Returns:
        train_df, val_df
    """
    train_df = df[df['fold'] != fold].reset_index(drop=True)
    val_df = df[df['fold'] == fold].reset_index(drop=True)

    return train_df, val_df
