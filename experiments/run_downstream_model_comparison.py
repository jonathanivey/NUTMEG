from models.multi_task import run_multi_task
import argparse
import random
import os
import pandas as pd


def main(IN_DIRECTORY,
         OUT_DIRECTORY,
         MODEL_SAVE_DIRECTORY,
         BASE_MODEL = "answerdotai/ModernBERT-base",
         ):

    # set seed
    random.seed(42)

    # set static parameter
    TRAINING_LOSS = VALIDATION_LOSS = "cross_entropy"


    for task in ['popquorn_politeness', 'popquorn_offensiveness']:

        print("\n\nRunning task:", task)

        # load in base dataframes for the task
        train_text_df = pd.read_csv(os.path.join(IN_DIRECTORY, task + "_train.csv")).drop_duplicates(['instance_id']).set_index('instance_id')[['text']]
        val_text_df = pd.read_csv(os.path.join(IN_DIRECTORY, task + "_val.csv")).drop_duplicates(['instance_id']).set_index('instance_id')[['text']]
        test_df = pd.read_csv(os.path.join(IN_DIRECTORY, task + "_test.csv"))

        # reduce the test_df to just instance_id and text
        test_df = test_df[['instance_id', 'text']].drop_duplicates()


        for method in  ["NUTMEG", "MACE", "majority", "majority_subpop", "individual"]:
            print("Running method:", method)

            if method == "NUTMEG" or method == "majority_subpop":
                if method == "majority_subpop":
                    method_x = "majority"
                else:
                    method_x = method

                for subpopulation in ['age', 'education', 'gender', 'race']:
                    # read in labels for the method
                    train_df = pd.read_csv(os.path.join(IN_DIRECTORY, task + "_train_" + method_x + "_" + subpopulation + ".csv"
                    )).drop_duplicates(['instance_id']).set_index('instance_id')
                    val_df = pd.read_csv(os.path.join(IN_DIRECTORY, task + "_val_" + method_x + "_" + subpopulation + ".csv"
                                                    )).drop_duplicates(['instance_id']).set_index('instance_id')

                    # join the labels to the text for our final train and val dataframes
                    train_df = train_text_df.join(train_df)
                    val_df = val_text_df.join(val_df)

                    # set which columns we're training on and how many labels (2)          
                    TRAINING_TASK_COLUMNS = list(train_df.columns[1:])
                    VALIDATION_TASK_COLUMNS = list(val_df.columns[1:])
                    NUM_LABELS = [2] * len(TRAINING_TASK_COLUMNS)

                    # set model save path
                    MODEL_SAVE_PATH = os.path.join(MODEL_SAVE_DIRECTORY, task, method_x + "_" + subpopulation)


                    # create / verify output directory
                    os.makedirs(os.path.join(OUT_DIRECTORY), exist_ok=True)

                    # create / verify model save directory
                    os.makedirs(os.path.join(MODEL_SAVE_DIRECTORY), exist_ok=True)
                    
                    # set predictions save path
                    PREDS_PATH = os.path.join(OUT_DIRECTORY, task + "_pred_" + method_x + "_" + subpopulation + ".csv")
                    
                    # run model
                    run_multi_task(train_df,
                            val_df,
                            test_df,
                            PREDS_PATH,
                            MODEL_SAVE_PATH,
                            TRAINING_LOSS,
                            VALIDATION_LOSS,
                            NUM_LABELS,
                            TRAINING_TASK_COLUMNS,
                            VALIDATION_TASK_COLUMNS,
                            BASE_MODEL)
                    
            else:
                # read in labels for the method
                train_df = pd.read_csv(os.path.join(IN_DIRECTORY, task + "_train_" + method + ".csv")).drop_duplicates(['instance_id']).set_index('instance_id')
                val_df = pd.read_csv(os.path.join(IN_DIRECTORY, task + "_val_" + method + ".csv")).drop_duplicates(['instance_id']).set_index('instance_id')

                # join the labels to the text for our final train and val dataframes
                train_df = train_text_df.join(train_df)
                val_df = val_text_df.join(val_df)

                # set which columns we're training on and how many labels (2)          
                TRAINING_TASK_COLUMNS = list(train_df.columns[1:])
                VALIDATION_TASK_COLUMNS = list(val_df.columns[1:])
                NUM_LABELS = [2] * len(TRAINING_TASK_COLUMNS)


                # set model save path
                MODEL_SAVE_PATH = os.path.join(MODEL_SAVE_DIRECTORY, task, method)

                # create / verify output directory
                os.makedirs(os.path.join(OUT_DIRECTORY), exist_ok=True)
                
                # create / verify model save directory
                os.makedirs(os.path.join(MODEL_SAVE_DIRECTORY), exist_ok=True)

                # set predictions save path
                PREDS_PATH = os.path.join(OUT_DIRECTORY, task + "_pred_" + method + ".csv")

                # run model
                run_multi_task(train_df,
                                        val_df,
                                        test_df,
                                        PREDS_PATH,
                                        MODEL_SAVE_PATH,
                                        TRAINING_LOSS,
                                        VALIDATION_LOSS,
                                        NUM_LABELS,
                                        TRAINING_TASK_COLUMNS,
                                        VALIDATION_TASK_COLUMNS,
                                        BASE_MODEL)
            print("Run Complete!")
            

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train models and make predictions with data from MACE, NUTMEG, majority vote, and non-aggregation for all tasks.")
    parser.add_argument("IN_DIRECTORY", type=str, default="../data/training_data/",help="Path to directory with train/val/test data for all tasks and models.")
    parser.add_argument("OUT_DIRECTORY", type=str, default="", help="Path to directory where we output the model predictions on the test set.")
    parser.add_argument("MODEL_SAVE_DIRECTORY", type=str, default="../models/", help="Path to directory where we output the trained models and tokenizers.")
    parser.add_argument("--BASE_MODEL", type=str, default="answerdotai/ModernBERT-base", help="Pretrained language model that we fine-tune.")
    args = parser.parse_args()
    
    main(args.IN_DIRECTORY, args.OUT_DIRECTORY, args.MODEL_SAVE_DIRECTORY, args.BASE_MODEL)