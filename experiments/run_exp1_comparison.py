import itertools
import pandas as pd
from multiprocessing import Pool, cpu_count

from NUTMEG.mace import MACE
from NUTMEG.nutmeg import NUTMEG
import pandas as pd
from sklearn.metrics import accuracy_score

# single run of experiment
def run_spam_div_experiment(spam_rate, div_rate):

    print('Running with spam:', spam_rate, ' and div:', div_rate)

    truth_df = pd.read_csv('/shared/3/projects/annotator-disagreement/data/synth_data/synth_truths_' + str(spam_rate) + '_' + str(div_rate) + '.csv')
    synth_df = pd.read_csv('/shared/3/projects/annotator-disagreement/data/synth_data/synth_annotations_' + str(spam_rate) + '_' + str(div_rate) + '.csv')

    bcc_labels = truth_df['task'].map(pd.read_csv('/shared/3/projects/annotator-disagreement/data/synth_data/zheng_out_BCC' + str(spam_rate) + '_' + str(div_rate) + '.csv').set_index('question')['result'])
    lfc_labels = truth_df['task'].map(pd.read_csv('/shared/3/projects/annotator-disagreement/data/synth_data/zheng_out_LFCbinary' + str(spam_rate) + '_' + str(div_rate) + '.csv').set_index('question')['result'])
    ds_labels = truth_df['task'].map(pd.read_csv('/shared/3/projects/annotator-disagreement/data/synth_data/zheng_out_EM' + str(spam_rate) + '_' + str(div_rate) + '.csv').set_index('question')['result'])
    majority_labels = truth_df['task'].map(synth_df.groupby('task')['label'].mean().round().astype(int))

    # initialize models
    nutmeg = NUTMEG(n_iter=1000)
    mace = MACE(n_iter=1000)

    # run MACE and NUTMEG on synthetic data
    mace.fit_predict(synth_df)
    nutmeg.fit_predict(synth_df)

    # store our experiment results in a dataframe row
    temp_df = pd.DataFrame({
        'method': ['Majority Vote', 'D&S', 'BCC', 'LFC', 'MACE', 'NUTMEG'],
        'spamming_rate': [spam_rate] * 6,
        'divisiveness_rate': [div_rate] * 6,
        'majority_accuracy': [accuracy_score(majority_labels, truth_df['majority_truth']),
                            accuracy_score(ds_labels, truth_df['majority_truth']),
                            accuracy_score(bcc_labels, truth_df['majority_truth']),
                            accuracy_score(lfc_labels, truth_df['majority_truth']),
                            accuracy_score(mace.labels_, truth_df['majority_truth']),
                            accuracy_score(nutmeg.labels_[:, 0].astype(int), truth_df['majority_truth'])],
        'minority_accuracy': [accuracy_score(majority_labels, truth_df['minority_truth']),
                            accuracy_score(ds_labels, truth_df['minority_truth']),
                            accuracy_score(bcc_labels, truth_df['minority_truth']),
                            accuracy_score(lfc_labels, truth_df['minority_truth']),
                            accuracy_score(mace.labels_, truth_df['minority_truth']),
                            accuracy_score(nutmeg.labels_[:, 1].astype(int), truth_df['minority_truth'])]
    })

    return temp_df

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
    experiment_df = pd.concat(results, ignore_index=True)
    
    experiment_df.to_csv('./results/figure_data/exp1_comparison.csv', index=False)

if __name__ == "__main__":
    main() 