"""
Step 7 (part 2): split the full file list into train/validation/test sets.

Split strategy and its limitation (documented honestly, see docs/dataset.md):
we stratify by class label only (70/15/15, fixed random seed for
reproducibility). Ideally recordings from the same baby/session would never
span across splits, to avoid the model "memorizing" one baby's voice rather
than learning general cry characteristics. However, the raw dataset's
data.csv only provides (is_cry, filename) - no speaker/session ID - so a
proper session-aware split isn't possible with the metadata we have. This is
listed as a known limitation, not silently ignored.
"""
from sklearn.model_selection import train_test_split

RANDOM_SEED = 42


def stratified_split(file_label_pairs, train_frac=0.70, val_frac=0.15):
    """
    Split a list of (file_path, label) tuples into train/val/test lists,
    preserving class proportions in each split.
    """
    labels = [label for _, label in file_label_pairs]

    train, remainder = train_test_split(
        file_label_pairs,
        train_size=train_frac,
        stratify=labels,
        random_state=RANDOM_SEED,
    )

    remainder_labels = [label for _, label in remainder]
    # remaining fraction is split evenly between val/test (e.g. 0.15/0.30 = 0.5)
    val_frac_of_remainder = val_frac / (1 - train_frac)
    val, test = train_test_split(
        remainder,
        train_size=val_frac_of_remainder,
        stratify=remainder_labels,
        random_state=RANDOM_SEED,
    )

    return train, val, test
