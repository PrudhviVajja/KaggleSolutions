# Import the libraries
import pandas as pd
from sklearn import model_selection

if __name__  == "__main__":

    # Load data
    df = pd.read_csv("/Users/pvajja/Kaggle/SetUp/input/train.csv")
    # Assign a new kfold column to "-1"
    df["kfold"] = -1

    # Shuffle Data
    df = df.sample(frac =1).reset_index(drop = True)

    # Define a Stratified K fold model 
    kf = model_selection.StratifiedKFold(n_splits=5, shuffle=False, random_state=42)

    # Loop over all the fold indexes
    for fold, (train_idx, val_idx) in enumerate(kf.split(X = df, y = df.target.values)):
        print(len(train_idx), len(val_idx))
        # Change the "kfold" column values to their respetive fold numbers
        df.loc[val_idx, "kfold"] = fold
    
    # Create a folds file in te input path:
    df.to_csv("/Users/pvajja/Kaggle/SetUp/input/train_folds.csv", index = False)

