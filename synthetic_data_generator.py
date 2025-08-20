import pandas as pd
import numpy as np
from collections import defaultdict
from sklearn.utils import shuffle
import random


def generate_synth_data(num_items, num_labels, annotators_per_item, max_items_per_annotator, subpopulation_proportions, 
                        expected_spamming_rate, divisiveness_rate, individual_variability, return_truths=False):

    # create dataframe to keep track of item information
    item_df = pd.DataFrame(index=list(range(num_items)))

    # create a dataframe to keep track of annotators information
    num_annotators =  (num_items * (annotators_per_item + 1) ) // max_items_per_annotator
    annotator_df = pd.DataFrame(index=list(range(num_annotators)))

    # generate empty arrays to fill with annotations, spamming indicators, and individual variance indicators
    # each row is an item, each column is an annotator
    annotations = np.empty((num_items, num_annotators))
    spammings = np.empty((num_items, num_annotators))
    individual_varyings = np.empty((num_items, num_annotators))

    ## generate subpopulations based on our predefined proportions

    # ensure our proportions add to 1
    subpop_total = sum(subpopulation_proportions.values())
    subpopulation_proportions = {key:(value / subpop_total) for key, value in subpopulation_proportions.items()}


    # calculate number of annotators with this subpopulation
    proportions = np.array(list(subpopulation_proportions.values()))
    subpop_counts = np.ceil(proportions * num_annotators).astype(int)

    # adjust our counts to be exactly num_annotators
    while subpop_counts.sum() != num_annotators:
        subpop_counts[np.argmax(subpop_counts)] -= 1

    # turn counts into dictionary
    subpop_counts = dict(zip(subpopulation_proportions.keys(), subpop_counts))

    # add subpopulations to dataframe
    annotator_df['subpopulation'] = np.concatenate([np.full(count, subpop) for subpop, count in subpop_counts.items()])


    ## generate spamming rate for each annotator

    # ensure our expected spamming rates are not exactly 0 or 1 to avoid issues with beta distribution being undefined
    for key, rate in expected_spamming_rate.items():
        if rate <= 0:
            expected_spamming_rate[key] = 1e-10
        elif rate >= 1:
            expected_spamming_rate[key] = 1-1e-10


    spamming_rates = np.array([])

    for subpop, count in subpop_counts.items():
        # sample our spamming rate from a beta distribution to account for the fact that
        # most people either spam a lot or spam rarely
        # we choose alpha = expected rate and beta = 1 - expected rate so the E(X) = expected rate
        spamming_rates = np.append(spamming_rates, (np.random.beta(expected_spamming_rate[subpop], 1-expected_spamming_rate[subpop], size=count)))

    # add spamming rates to dataframe
    annotator_df['spamming_rate'] = spamming_rates

    # add individual variabilities to dataframe
    annotator_df['individual_variability'] = annotator_df['subpopulation'].map(individual_variability)

    # randomize the annotators
    annotator_df = shuffle(annotator_df).reset_index(drop=True)

    ## generate true values for each item

    for item in range(num_items):

        # if this is a divisive item
        if np.random.binomial(1, divisiveness_rate) == 1:

            # randomly select labels for each subpopulation category
            if len(subpopulation_proportions) <= num_labels:
                true_labels = np.random.choice(num_labels, len(subpopulation_proportions), replace=False)
            else:
                true_labels = np.random.choice(num_labels, len(subpopulation_proportions), replace=True)

            # add true labels to item dataframe
            for i, subpop in enumerate(subpopulation_proportions.keys()):
                item_df.loc[item, str(subpop) + '_truth'] = true_labels[i]
            
        else:
            # if not divergent, choose a single label randomly 
            true_label = np.random.randint(num_labels)

            # add true labels to item dataframe
            for i, subpop in enumerate(subpopulation_proportions.keys()):
                item_df.loc[item, str(subpop) + '_truth'] = true_label


    ## generate annotation for each annotator for each item
    for item in range(num_items):
        # iterate through each annotator to decide what their label is        
        for annotator in range(num_annotators):

            # if annotator is spamming
            if np.random.binomial(1, annotator_df.loc[annotator, 'spamming_rate']) == 1:

                # choose our label from a discrete uniform distribution
                annotations[item][annotator] = np.random.randint(num_labels)

                # mark this label as spam
                spammings[item][annotator] = 1

            else:
                
                # mark this label as not spam
                spammings[item][annotator] = 0

                # if this individual is varying
                if np.random.binomial(1, annotator_df.loc[annotator, 'individual_variability']) == 1:
                    # set our label randomly
                    annotations[item][annotator] = np.random.randint(num_labels)

                    # mark this label as individual variance
                    individual_varyings[item][annotator] = 1

                else: 
                    # mark this  as not individual variance
                    individual_varyings[item][annotator] = 0

                    # set our label based on subpopulation
                    annotations[item][annotator] = item_df.loc[item, str(annotator_df.loc[annotator, 'subpopulation']) + '_truth']


    ## decide which annotators annotations will be available for each item

    # intialize empty dataframe to store synthetic data
    synth_df = pd.DataFrame()

    # intiailize dict to keep track of which annotators are labeling which items
    annotator_items = defaultdict(list)


    for item in range(num_items):
        # look at list of available annotators
        available_annotators = [a for a in list(range(num_annotators)) if len(annotator_items[a]) < max_items_per_annotator]

        # check to ensure that our calculations are possible
        if len(available_annotators) < annotators_per_item:
            raise ValueError("It's not possible to assign the required number of annotators per item with the given constraints.")

        # randomly select which annotator goes with this item
        selected_annotators = random.sample(available_annotators, annotators_per_item)

        # for each selected annoatotor, add row to synthetic dataframe
        for annotator in selected_annotators:
            df_row = {**{'task':item,
                        'annotator':annotator,
                        'subpopulation': annotator_df.loc[annotator, 'subpopulation'],
                        'label':annotations[item][annotator],
                        'spamming':spammings[item][annotator],
                        'individual_varying':individual_varyings[item][annotator],
                        'spamming_rate':annotator_df.loc[annotator, 'spamming_rate'],
                        'individual_variability': annotator_df.loc[annotator, 'individual_variability']},
                        **{str(subpop) + '_truth':item_df.loc[item, str(subpop) + '_truth'] for subpop in subpopulation_proportions.keys()}}


            synth_df = synth_df._append(df_row, ignore_index=True)

            # mark this annotator as having done this item
            annotator_items[annotator].append(item)

    
    return synth_df