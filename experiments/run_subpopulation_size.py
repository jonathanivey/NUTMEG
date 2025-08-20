import itertools
import pandas as pd
from multiprocessing import Pool, cpu_count

from NUTMEG.mace import MACE
from NUTMEG.nutmeg import NUTMEG
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from synthetic_data_generator import generate_synth_data


# single run of experiment
def run_subpop_size_experiment(subpop_size, annotators_per_item):
    print(str(subpop_size) +'-' +str(annotators_per_item), 'Running with subpop size:', subpop_size, 'and', annotators_per_item, 'annotations per item')

    # intialize models
    nutmeg = NUTMEG(n_iter=1000)
    mace = MACE(n_iter=1000)

    # set conditions
    num_items = 500
    num_labels = 2
    max_items_per_annotator = 100

    # what proportion of the population is each subpopulation
    subpopulation_proportions = {'primary':subpop_size, 'secondary':(1-subpop_size)}

    # how often do the different segments of the population spam (Expected value of our beta distribution)
    expected_spamming_rate = {'primary':0.1, 'secondary':0.1}

    # what proportion of our items will be divisive in some way
    divisiveness_rate = 0.2

    # how often to the genuine opinions of each annotator vary based on demographics
    individual_variability= {'primary':0, 'secondary':0}

    # generate our synthetic data
    print(str(subpop_size) +'-' +str(annotators_per_item),'Generating data ...')
    synth_df = generate_synth_data(num_items, num_labels, annotators_per_item, max_items_per_annotator, subpopulation_proportions, 
                        expected_spamming_rate, divisiveness_rate, individual_variability, return_truths=True)

    # sort by subpopulation, so that the first subpopulation encountered is always the majority
    synth_df = synth_df.sort_values("subpopulation")
    synth_df = synth_df.rename(columns={'annotator':'worker'})
    synth_df['label'] = synth_df['label'].astype(str)

    # create a dataframe so we can compare the real data to the predictions
    truth_df = synth_df[['task', 'primary_truth']].drop_duplicates()

    # run MACE and NUTMEG on synthetic data
    print(str(subpop_size) +'-' +str(annotators_per_item),'Running MACE ...')
    mace.fit_predict(synth_df)
    print(str(subpop_size) +'-' +str(annotators_per_item),'Running NUTMEG ...')
    nutmeg.fit_predict(synth_df)

    print(str(subpop_size) +'-' +str(annotators_per_item),'Saving dataframe ...')

    # store our experiment results in a dataframe row
    temp_df = pd.DataFrame({
        'method': ['MACE', 'NUTMEG'],
        'subpopulation_size': subpop_size,
        'annotations_per_item': annotators_per_item,
        'accuracy': [accuracy_score(mace.labels_, truth_df['primary_truth']), accuracy_score(nutmeg.labels_[:, 0], truth_df['primary_truth'])],
        'f1': [f1_score(mace.labels_, truth_df['primary_truth']), f1_score(nutmeg.labels_[:, 0], truth_df['primary_truth'])]

    })


    return temp_df

# wrapper function for multiprocessing
def experiment_wrapper(args):
    subpop_size, annotators_per_item = args
    return run_subpop_size_experiment(subpop_size, annotators_per_item)

def main():
    # define different subpopulation size and number of annotators to compare
    subpop_sizes = [0.1, 0.2, 0.3, 0.4, 0.5]
    annotators_numbers = [3, 5, 7, 9, 11, 13, 15]
    
    # generate all combinations of subpopulation size and annotators_per_item
    combinations = list(itertools.product(subpop_sizes, annotators_numbers))
    
    # sue multiprocessing pool
    with Pool(round(cpu_count() * 0.3)) as pool:
        results = pool.map(experiment_wrapper, combinations)
    
    print("Combining Dataframes ...")

    # combine all dataframes into one
    experiment_df = pd.concat(results, ignore_index=True)
    

    print("Saving to file ...")
    experiment_df.to_csv('./results/figure_data/subpopulation_size.csv', index=False)

    print("Complete")

if __name__ == "__main__":
    main()