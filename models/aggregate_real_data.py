import os
import sys
# Get the absolute path to the directory
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_dir)

import pandas as pd
from multiprocessing import Pool, cpu_count
import itertools
import argparse
from NUTMEG.mace import MACE
from NUTMEG.nutmeg import NUTMEG
import pandas as pd

def mace_aggregate(task, split, DIRECTORY):
    mace = MACE(n_iter=1000)
    
    df = pd.read_csv(os.path.join(DIRECTORY, task + "_" + split + ".csv"))
    df = df.rename(columns={'instance_id':'task', 'user_id':'worker', 'binary_label':'label'})
    df['label'] = df['label'].astype(float)

    print('Running MACE')
    mace.fit_predict(df)

    OUT_DIRECTORY = os.path.join(DIRECTORY, task + "_" + split + "_MACE.csv")

    print('Saving MACE predictions to', OUT_DIRECTORY)
    df['label'] = df['task'].map(mace.labels_)

    df = df.rename(columns={'task':'instance_id'})

    df[['instance_id', 'label']].to_csv(OUT_DIRECTORY, index=False)

def nutmeg_aggregate(task, subpopulation, split, DIRECTORY):
    nutmeg = NUTMEG(n_iter=1000)
    
    df = pd.read_csv(os.path.join(DIRECTORY, task + "_" + split + ".csv"))
    df = df.rename(columns={'instance_id':'task', 'user_id':'worker', 'binary_label':'label', subpopulation:'subpopulation'})
    df['label'] = df['label'].astype(float)

    print('Running NUTMEG')
    nutmeg.fit_predict(df, return_unobserved=False)

    OUT_DIRECTORY = os.path.join(DIRECTORY, task + "_" + split + "_NUTMEG_" + subpopulation + ".csv")

    print('Saving NUTMEG predictions to', OUT_DIRECTORY)
    label_cols = []

    for i, subpop_value in enumerate(df['subpopulation'].unique()):
        df[str(subpop_value) + '_label'] = df['task'].map(dict(zip(df['task'].unique(), nutmeg.labels_[:, i])))
        label_cols.append(str(subpop_value) + '_label')

    df = df.rename(columns={'task':'instance_id'})

    df[['instance_id'] + label_cols].to_csv(OUT_DIRECTORY, index=False)



def majority_aggregate(task, split, DIRECTORY):

    print('Running majority')
    
    df = pd.read_csv(os.path.join(DIRECTORY, task + "_" + split + ".csv"))
    df['binary_label'] = df['binary_label'].astype(float)

    df['label'] = df['instance_id'].map(df.groupby('instance_id')['binary_label'].mean()).round()

    OUT_DIRECTORY = os.path.join(DIRECTORY, task + "_" + split + "_majority.csv")

    print('Saving majority predictions to', OUT_DIRECTORY)

    df[['instance_id', 'label']].to_csv(OUT_DIRECTORY, index=False)

def majority_subpop_aggregate(task, subpopulation, split, DIRECTORY):

    print('Running majority by subpopulation')
    
    df = pd.read_csv(os.path.join(DIRECTORY, task + "_" + split + ".csv"))
    df['binary_label'] = df['binary_label'].astype(float)

    grouped_df = df.groupby(['instance_id', subpopulation])['binary_label'].mean().round().reset_index()

    OUT_DIRECTORY = os.path.join(DIRECTORY, task + "_" + split + "_majority_" + subpopulation + ".csv")

    print('Saving majority predictions to', OUT_DIRECTORY)

    grouped_df.pivot_table(index='instance_id', columns=subpopulation, values='binary_label').to_csv(OUT_DIRECTORY, index=True)



# wrapper function for multiprocessing
def experiment_wrapper(args):
    model, task, subpopulation, split, directory = args

    if model == 'MACE':
        mace_aggregate(task, split, directory)
    elif model == 'NUTMEG':
        nutmeg_aggregate(task, subpopulation, split, directory)
    elif model == "majority":
        majority_aggregate(task, split, directory)
    elif model == "majority_subpop":
        majority_subpop_aggregate(task, subpopulation, split, directory)
    

def main(DIRECTORY):

    splits = ['train', 'val']
    tasks = ['popquorn_politeness', 'popquorn_offensiveness']
    subpopulations = ['age', 'education', 'gender', 'race']
    
    # generate all combinations of our values of interest
    nutmeg_combinations = list(itertools.product(['NUTMEG', 'majority_subpop'], tasks, subpopulations, splits, [DIRECTORY]))
    other_combinations = list(itertools.product(['MACE', 'majority',], tasks, [''], splits, [DIRECTORY]))
    combinations = nutmeg_combinations + other_combinations

    # use multiprocessing Pool
    with Pool(cpu_count()) as pool:
        pool.map(experiment_wrapper, combinations)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run MACE, NUTMEG, majority vote, and non-aggregation on all tasks.")
    parser.add_argument("DATA_PATH", type=str, default="../data/training_data/", help="Path to directory with training data for all tasks.")
    args = parser.parse_args()
    
    main(args.DATA_PATH)