import itertools
import pandas as pd
from multiprocessing import Pool, cpu_count

from NUTMEG.nutmeg import NUTMEG
from NUTMEG.mace import MACE
import pandas as pd
import numpy as np
from synthetic_data_generator import generate_synth_data

# single run of experiment
def run_competence_correlation_experiment(spam_rate, div_rate):
    print('Running with spam:', spam_rate, ' and div:', div_rate)

    # initialize models
    nutmeg = NUTMEG(n_iter=1000)
    mace = MACE(n_iter=1000)

    # set conditions
    num_items = 500
    num_labels = 2
    annotators_per_item = 5
    max_items_per_annotator = 20

    # what proportion of the population is each demographic
    subpopulation_proportions = {'majority':0.8, 'minority':0.2}

    # how often do the different segments of the population spam (Expected value of our beta distribution)
    expected_spamming_rate = {'majority':spam_rate, 'minority':spam_rate}

    # what proportion of our items will be divisive in some way
    divisiveness_rate = div_rate

    # how often to the genuine opinions of each annotator vary based on demographics
    individual_variability= {'majority':0, 'minority':0}

    print(spam_rate, div_rate, 'Generating data ...')
    # generate our synthetic data
    synth_df = generate_synth_data(num_items, num_labels, annotators_per_item, max_items_per_annotator, subpopulation_proportions, 
                        expected_spamming_rate, divisiveness_rate, individual_variability, return_truths=True)

    # sort by subpopulation, so that the first subpopulation encountered is always the majority
    synth_df = synth_df.sort_values("subpopulation")
    synth_df = synth_df.rename(columns={'annotator':'worker'})
    synth_df['label'] = synth_df['label'].astype(str)

    # run MACE and NUTMEG on synthetic data
    print(spam_rate, div_rate, 'Running MACE ...')
    mace.fit_predict(synth_df)

    print(spam_rate, div_rate, 'Running NUTMEG ...')
    nutmeg.fit_predict(synth_df, return_unobserved=True)

    # get estimated competence ratings
    nutmeg_competences = nutmeg.spamming_[:, 0]
    mace_competences = mace.spamming_[:, 0]

    # get true competence ratings from synthetic data
    true_competences = np.array(synth_df.groupby('worker')['spamming_rate'].mean()[synth_df['worker'].unique()])

    # identify which workers are majority vs minority
    majority_indices = (synth_df.groupby('worker')['subpopulation'].max() == 'majority')
    minority_indices = (synth_df.groupby('worker')['subpopulation'].max() == 'minority')

    # calculate correlations
    nutmeg_correlation = np.corrcoef(true_competences, nutmeg_competences)[0, 1]
    mace_correlation = np.corrcoef(true_competences, mace_competences)[0, 1]

    nutmeg_majority_correlation = np.corrcoef(true_competences[majority_indices], nutmeg_competences[majority_indices])[0, 1]
    mace_majority_correlation = np.corrcoef(true_competences[majority_indices], mace_competences[majority_indices])[0, 1]

    nutmeg_minority_correlation = np.corrcoef(true_competences[minority_indices], nutmeg_competences[minority_indices])[0, 1]
    mace_minority_correlation = np.corrcoef(true_competences[minority_indices], mace_competences[minority_indices])[0, 1]

    # store our experiment results in a dataframe row
    temp_df = pd.DataFrame({
        'method': ['MACE', 'NUTMEG'],
        'spamming_rate': [spam_rate, spam_rate],
        'divisiveness_rate': [div_rate, div_rate],
        'competence_correlation': [mace_correlation, nutmeg_correlation],
        'majority_competence_correlation': [mace_majority_correlation, nutmeg_majority_correlation],
        'minority_competence_correlation': [mace_minority_correlation, nutmeg_minority_correlation],

    })

    return temp_df

# wrapper function for multiprocessing
def experiment_wrapper(args):
    spam_rate, div_rate = args
    return run_competence_correlation_experiment(spam_rate, div_rate)

def main():
    # define different spamming and divisiveness rates to compare
    spam_rates = [0, 0.05, 0.1, 0.15, 0.2, 0.25]
    div_rates = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    
    # Generate all combinations of spam_rate and div_rate
    combinations = list(itertools.product(spam_rates, div_rates))
    
    # Use multiprocessing Pool
    with Pool(cpu_count()) as pool:
        results = pool.map(experiment_wrapper, combinations)
    
    # Combine all DataFrames into one
    experiment_df = pd.concat(results, ignore_index=True)
    
    experiment_df.to_csv('./results/figure_data/competence_correlation.csv', index=False)

if __name__ == "__main__":
    main()