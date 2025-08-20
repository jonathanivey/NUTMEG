import argparse
import os
import pandas as pd

def process_politeness(RAW_DATA_PATH, OUT_DIRECTORY, SPLITS_DIRECTORY="./splits/"):
    df = pd.read_csv(RAW_DATA_PATH)

    # create binary classification labels
    df['binary_label'] = (df['politeness'] >= 3).astype(int)


    # Remap some age groups
    age_mapping = {
        "18-24": "18-29", "25-29": "18-29",
        "30-34": "30-39", "35-39": "30-39",
        "40-44": "40-49", "45-49": "40-49",
        "50-54":"50-59", "54-59":"50-59",
        "60-64":"60+", ">65":"60+"
    }

    df["age"] = df["age"].replace(age_mapping)

    # drop all unkown age values
    df = df[df["age"] != "Prefer not to disclose"]

    # filter to genders with > 0.05
    df = df[df['gender'].isin(["Woman", "Man"])]

    # filter to educations with > 0.05
    df = df[df['education'].isin(["College degree", "High school diploma or equivalent", "Graduate degree"])]

    # filter to races with > 0.05
    df = df[df['race'].isin(["White", "Black or African American", "Hispanic or Latino", "Asian"])]

    # create distribution columns for politeness
    for demographic in ['gender', 'race', 'age', 'education']:
        # Compute the distribution
        distribution_df = (
            df.groupby(["instance_id", demographic])["binary_label"]
            .value_counts(normalize=True)  # Get proportion
            .unstack(fill_value=0)  # Pivot the binary label values into separate columns
        )

        # Convert to list format
        distribution_df = distribution_df.apply(lambda row: [row[0], row[1]], axis=1).unstack()

        # Reset index and format the final DataFrame
        distribution_df.columns.name = None
        distribution_df = distribution_df.reset_index()
        distribution_df = distribution_df.rename(columns={column:column+"_distribution_label" for column in distribution_df.columns[1:]})

        df = pd.merge(df, distribution_df, on='instance_id', how='left')

        # read in politeness train-val-test splits
        train_split = pd.read_csv(SPLITS_DIRECTORY+ 'politeness_train_split')['instance_id']
        val_split = pd.read_csv(SPLITS_DIRECTORY + 'politeness_val_split')['instance_id']
        test_split = pd.read_csv(SPLITS_DIRECTORY + 'politeness_test_split')['instance_id']

        # save training, val, and test sets for politeness
        df[df['instance_id'].isin(train_split)].to_csv(OUT_DIRECTORY + "popquorn_politeness_train.csv", index=False)
        df[df['instance_id'].isin(val_split)].to_csv(OUT_DIRECTORY + "popquorn_politeness_val.csv", index=False)
        df[df['instance_id'].isin(test_split)].to_csv(OUT_DIRECTORY + "popquorn_politeness_test.csv", index=False)

    # save individualized versions of train and validation sets
    df[df['instance_id'].isin(train_split)].pivot(
        index="instance_id", columns="user_id", values="binary_label").reset_index().to_csv(OUT_DIRECTORY + "popquorn_politeness_train_individual.csv", index=False)
    df[df['instance_id'].isin(val_split)].pivot(
        index="instance_id", columns="user_id", values="binary_label").reset_index().to_csv(OUT_DIRECTORY + "popquorn_politeness_val_individual.csv", index=False)
        


def process_offensiveness(RAW_DATA_PATH, OUT_DIRECTORY, SPLITS_DIRECTORY="./splits/"):
    df = pd.read_csv(RAW_DATA_PATH)

    # create binary classification labels
    df['binary_label'] = (df['offensiveness'] >= 3).astype(int)


    # Remap some age groups
    age_mapping = {
        "18-24": "18-29", "25-29": "18-29",
        "30-34": "30-39", "35-39": "30-39",
        "40-44": "40-49", "45-49": "40-49",
        "50-54":"50-59", "54-59":"50-59",
        "60-64":"60+", ">65":"60+"
    }

    df["age"] = df["age"].replace(age_mapping)

    # drop all unkown age values
    df = df[df["age"] != "Prefer not to disclose"]

    # filter to genders with > 0.05
    df = df[df['gender'].isin(["Woman", "Man"])]

    # filter to educations with > 0.05
    df = df[df['education'].isin(["College degree", "High school diploma or equivalent", "Graduate degree"])]

    # filter to races with > 0.05
    df = df[df['race'].isin(["White", "Black or African American", "Asian"])]

    # create distribution columns for offensiveness
    for demographic in ['gender', 'race', 'age', 'education']:
        # Compute the distribution
        distribution_df = (
            df.groupby(["instance_id", demographic])["binary_label"]
            .value_counts(normalize=True)  # Get proportion
            .unstack(fill_value=0)  # Pivot the binary label values into separate columns
        )

        # Convert to list format
        distribution_df = distribution_df.apply(lambda row: [row[0], row[1]], axis=1).unstack()

        # Reset index and format the final DataFrame
        distribution_df.columns.name = None
        distribution_df = distribution_df.reset_index()
        distribution_df = distribution_df.rename(columns={column:column+"_distribution_label" for column in distribution_df.columns[1:]})

        df = pd.merge(df, distribution_df, on='instance_id', how='left')

        # read in offensiveness train-val-test splits
        train_split = pd.read_csv(SPLITS_DIRECTORY+ 'offensiveness_train_split')['instance_id']
        val_split = pd.read_csv(SPLITS_DIRECTORY + 'offensiveness_val_split')['instance_id']
        test_split = pd.read_csv(SPLITS_DIRECTORY + 'offensiveness_test_split')['instance_id']

        # save training, val, and test sets for offensiveness
        df[df['instance_id'].isin(train_split)].to_csv(OUT_DIRECTORY + "popquorn_offensiveness_train.csv", index=False)
        df[df['instance_id'].isin(val_split)].to_csv(OUT_DIRECTORY + "popquorn_offensiveness_val.csv", index=False)
        df[df['instance_id'].isin(test_split)].to_csv(OUT_DIRECTORY + "popquorn_offensiveness_test.csv", index=False)

    # save individualized versions of train and validation sets
    df[df['instance_id'].isin(train_split)].pivot(
        index="instance_id", columns="user_id", values="binary_label").reset_index().to_csv(OUT_DIRECTORY + "popquorn_offensiveness_train_individual.csv", index=False)
    df[df['instance_id'].isin(val_split)].pivot(
        index="instance_id", columns="user_id", values="binary_label").reset_index().to_csv(OUT_DIRECTORY + "popquorn_offensiveness_val_individual.csv", index=False)
        

def main(RAW_DATA_PATH: str, OUT_DIRECTORY: str, TASK: str, SPLITS_DIRECTORY: str = "./splits/"):
    
    # Ensure output directory exists
    os.makedirs(OUT_DIRECTORY, exist_ok=True)
    os.makedirs(SPLITS_DIRECTORY, exist_ok=True)
    
    print(f"Preprocessing {RAW_DATA_PATH} for task: {TASK}")
    print(f"Output will be saved to {OUT_DIRECTORY}")
    print(f"Using splits directory: {SPLITS_DIRECTORY}")
    
    
    if TASK == "politeness":
        process_politeness(RAW_DATA_PATH=RAW_DATA_PATH, OUT_DIRECTORY=OUT_DIRECTORY, SPLITS_DIRECTORY=SPLITS_DIRECTORY)
    elif TASK == "offensiveness":
        process_offensiveness(RAW_DATA_PATH=RAW_DATA_PATH, OUT_DIRECTORY=OUT_DIRECTORY, SPLITS_DIRECTORY=SPLITS_DIRECTORY)
    else:
        raise ValueError("TASK must be either 'politeness' or 'offensiveness'")

    print("Preprocessing complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process raw data for politeness or offensiveness tasks.")
    parser.add_argument("TASK", type=str, choices=["politeness", "offensiveness"], help="Task type: politeness or offensiveness")
    parser.add_argument("RAW_DATA_PATH", type=str, help="Path to the raw CSV data file")
    parser.add_argument("OUT_DIRECTORY", type=str, default="./training_data/", help="Path to the output directory")
    parser.add_argument("--SPLITS_DIRECTORY", type=str, default="./splits/", help="Path to the splits directory")
    
    args = parser.parse_args()
    
    main(args.RAW_DATA_PATH, os.path.join(args.OUT_DIRECTORY, ''), args.TASK, os.path.join(args.SPLITS_DIRECTORY, ''))
