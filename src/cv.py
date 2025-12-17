import pandas as pd
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


def create_stratification_groups(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create stratification groups based on State and Species.
    Add season information for analysis.

    Args:
        df: DataFrame with 'Sampling_Date', 'State', 'Species' columns

    Returns:
        DataFrame with added 'Season' and 'strat_group' columns
    """
    df = df.copy()
    
    # Add season column for analysis
    df['Season'] = df['Sampling_Date'].apply(get_season)
    
    # Create stratification column: State + Species
    # This ensures both State and Species are balanced across folds
    df['strat_group'] = (
        df['State'].astype(str) + '_' +
        df['Species'].astype(str)
    )
    
    return df


def create_folds_with_date_groups(
    df: pd.DataFrame,
    n_folds: int = 5,
    random_state: int = 42,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Create folds using StratifiedGroupKFold.
    - Groups by Sampling_Date (prevents date leakage)
    - Stratifies by State + Species (balances distributions)

    Args:
        df: DataFrame with 'Sampling_Date', 'State', 'Species' columns
        n_folds: Number of folds
        random_state: Random seed for reproducibility
        verbose: Print fold statistics

    Returns:
        DataFrame with added 'fold' column
    """
    df = create_stratification_groups(df)
    
    # Use StratifiedGroupKFold
    # groups: Sampling_Date (prevents same date in train/val)
    # stratify: State + Species (balances across folds)
    sgkf = StratifiedGroupKFold(
        n_splits=n_folds,
        shuffle=True,
        random_state=random_state
    )
    
    # Initialize fold column
    df['fold'] = -1
    
    # Get groups (Sampling_Date) and stratification labels (State_Species)
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
                print(f"  ✓ No date overlap (train: {len(train_dates)} dates, val: {len(val_dates)} dates)")
    
    if verbose:
        verify_fold_quality(df, n_folds)
    
    return df


def verify_fold_quality(df: pd.DataFrame, n_folds: int) -> None:
    """
    Verify that folds have balanced distributions.

    Args:
        df: DataFrame with 'fold', 'Season', 'State', 'Species' columns
        n_folds: Number of folds
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
