import itertools
import pandas as pd
from multiprocessing import Pool, cpu_count

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from synthetic_data_generator import generate_synth_data

# single run of experiment
def run_spam_div_experiment(spam_rate, div_rate):
    print('Running with spam:', spam_rate, ' and div:', div_rate)

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

    # generate our synthetic data
    synth_df = generate_synth_data(num_items, num_labels, annotators_per_item, max_items_per_annotator, subpopulation_proportions, 
                        expected_spamming_rate, divisiveness_rate, individual_variability, return_truths=True)

    # sort by subpopulation, so that the first subpopulation encountered is always the majority
    synth_df = synth_df.sort_values("subpopulation")
    synth_df = synth_df.rename(columns={'annotator':'worker'})
    synth_df['label'] = synth_df['label'].astype(int).astype(str)
    synth_df['majority_truth'] = synth_df['majority_truth'].astype(int).astype(str)
    synth_df['minority_truth'] = synth_df['minority_truth'].astype(int).astype(str)

    # create a dataframe so we can compare the real data to the predictions
    truth_df = synth_df[['task', 'majority_truth', 'minority_truth']].drop_duplicates()

    synth_df.to_csv('/shared/3/projects/annotator-disagreement/data/synth_data/synth_annotations_' + str(spam_rate) + '_' + str(div_rate) +'.csv', index=False)
    truth_df.to_csv('/shared/3/projects/annotator-disagreement/data/synth_data/synth_truths_' + str(spam_rate) + '_' + str(div_rate) +'.csv', index=False)

    synth_df[['task', 'worker', 'label']].rename(columns={'task':'question', 'label':'answer'}
        ).to_csv('/shared/3/projects/annotator-disagreement/data/synth_data/zheng_annotations_'
                 + str(spam_rate) + '_' + str(div_rate) +'.csv', index=False)

# wrapper function for multiprocessing
def experiment_wrapper(args):
    spam_rate, div_rate = args
    return run_spam_div_experiment(spam_rate, div_rate)

def main():
    # define different spamming and divisiveness rates to compare
    spam_rates = [0, 0.05, 0.1, 0.15, 0.2, 0.25]
    div_rates = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    
    # generate all combinations of spam_rate and div_rate
    combinations = list(itertools.product(spam_rates, div_rates))
    
    # use multiprocessing Pool
    with Pool(cpu_count()) as pool:
        results = pool.map(experiment_wrapper, combinations)
    
    # combine all dataframes into one
    # experiment_df = pd.concat(results, ignore_index=True)
    
    # experiment_df.to_csv('./results/figure_data/mace_nutmeg_comparison.csv', index=False)

if __name__ == "__main__":
    main()