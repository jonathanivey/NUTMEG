import optuna
import torch
import torch.nn as nn
import torch.optim as optim
from transformers import AutoModel, AutoTokenizer
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
from ast import literal_eval
from copy import deepcopy

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def run_multi_task(train_df,
                   val_df,
                   test_df,
                   PREDS_PATH,
                   MODEL_SAVE_PATH,
                   TRAINING_LOSS,
                   VALIDATION_LOSS,
                   NUM_LABELS,
                   TRAINING_TASK_COLUMNS,
                   VALIDATION_TASK_COLUMNS,
                   BASE_MODEL):
    


    def parse_list(value):
        if pd.isna(value) or value.strip() == "":
            return np.nan
        return literal_eval(value)


    def load_data(df, tokenizer, max_length=192):
        """ Tokenizes text and prepares labels for multi-task learning. """
        encodings = tokenizer(list(df['text']), truncation=True, padding=True, max_length=max_length, return_tensors="pt")

        if df.shape[1] > 1:
            labels = {}


            if df.iloc[:,1].dtype == 'O':
                first_list = next(x for x in df.iloc[:,1] if isinstance(x, list))
                label_size = len(first_list)
                for i, task in enumerate(df.columns[1:]):
                    labels[str(i)] = torch.tensor([[float(y) for y in x] if isinstance(x, list) else label_size * [-1] for x in df[task]])
            else:
                for i, task in enumerate(df.columns[1:]):
                    labels[str(i)] = torch.tensor([float(x) if not pd.isna(x) else -1 for x in df[task]])
            
            encodings['labels'] = labels
        return encodings


    class MultiTaskModel(nn.Module):
        def __init__(self, model_name):
            super(MultiTaskModel, self).__init__()
            self.encoder = AutoModel.from_pretrained(model_name)
            self.classifiers = nn.ModuleDict({
                str(task): nn.Linear(self.encoder.config.hidden_size, num_labels)
                for task, num_labels in enumerate(NUM_LABELS)
            })
        
        def forward(self, input_ids, attention_mask):
            outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
            pooled_output = outputs.last_hidden_state[:, 0, :]
            logits = {task: classifier(pooled_output) for task, classifier in self.classifiers.items()}
            return logits


    def jsd_loss(p, q):
        """Compute Jensen-Shannon Divergence Loss between two distributions."""
        m = 0.5 * (p + q)
        return 0.5 * (F.kl_div(F.log_softmax(p, dim=-1), m, reduction='batchmean') +
                    F.kl_div(F.log_softmax(q, dim=-1), m, reduction='batchmean'))


    def train_multi_task_model(train_df, val_df, model_name, lr, batch_size, num_epochs):
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        train_encodings = load_data(train_df, tokenizer)
        val_encodings = load_data(val_df, tokenizer)


        train_dataset = TensorDataset(train_encodings['input_ids'], train_encodings['attention_mask'], 
                                    *[train_encodings['labels'][str(task)] for task in range(len(TRAINING_TASK_COLUMNS))])
        val_dataset = TensorDataset(val_encodings['input_ids'], val_encodings['attention_mask'], 
                                    *[val_encodings['labels'][str(task)] for task in range(len(VALIDATION_TASK_COLUMNS))])

        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        model = MultiTaskModel(model_name).to(device)
        optimizer = optim.AdamW(model.parameters(), lr=lr)
        
        best_val_loss = float('inf')
        best_model_state = None
        
        for epoch in range(num_epochs):
            model.train()
            total_train_loss = 0
            for batch in train_dataloader:
                input_ids, attention_mask, *labels = [x.to(device) for x in batch]
                optimizer.zero_grad()
                logits = model(input_ids, attention_mask)
                
                loss = 0
                for task in range(len(TRAINING_TASK_COLUMNS)):
                    if TRAINING_LOSS == 'cross_entropy':
                        valid_mask = labels[task] != -1
                        if valid_mask.any():
                            criterion = nn.CrossEntropyLoss()
                            task_loss = criterion(logits[str(task)][valid_mask], labels[task][valid_mask].long())
                            loss += task_loss
                    elif TRAINING_LOSS == 'jsd':
                        valid_mask = ~(labels[task] == -1).all(dim=1)
                        if valid_mask.any():
                            task_loss = jsd_loss(F.softmax(logits[str(task)][valid_mask], dim=-1), labels[task][valid_mask])
                            loss += task_loss
                
                loss.backward()
                optimizer.step()
                total_train_loss += loss.item()
            
            print(f"Epoch {epoch+1}, Train Loss: {total_train_loss:.4f}")

            # Validation loss
            model.eval()
            total_val_loss = 0
            with torch.no_grad():
                for batch in val_dataloader:
                    input_ids, attention_mask, *labels = [x.to(device) for x in batch]
                    logits = model(input_ids, attention_mask)
                    
                    loss = 0
                    for task in range(len(VALIDATION_TASK_COLUMNS)):
                        if VALIDATION_LOSS == 'cross_entropy':
                            valid_mask = labels[task] != -1
                            if valid_mask.any():
                                criterion = nn.CrossEntropyLoss()
                                task_loss = criterion(logits[str(task)][valid_mask], labels[task][valid_mask].long())
                                loss += task_loss
                        elif VALIDATION_LOSS == 'jsd':
                            valid_mask = ~(labels[task] == -1).all(dim=1)
                            if valid_mask.any():
                                task_loss = jsd_loss(F.softmax(logits[str(task)][valid_mask], dim=-1), labels[task][valid_mask])
                                loss += task_loss
                    
                    total_val_loss += loss.item()
            
            print(f"Epoch {epoch+1}, Validation Loss: {total_val_loss:.4f}")
            
            # Save the best model
            if total_val_loss < best_val_loss:
                best_val_loss = total_val_loss
                best_model_state = deepcopy(model.state_dict())
        
        # Load best model state
        model.load_state_dict(best_model_state)
        
        return model, tokenizer, best_val_loss



    def predict(model, tokenizer, test_df, batch_size=64):
        model.eval()
        encodings = load_data(test_df, tokenizer)
        test_dataset = TensorDataset(encodings['input_ids'], encodings['attention_mask'])
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        all_predictions = []
        with torch.no_grad():
            for batch in test_dataloader:
                input_ids, attention_mask = [x.to(device) for x in batch]
                logits = model(input_ids, attention_mask)
                
                batch_predictions = {
                    task: F.softmax(logits[task], dim=-1).cpu().numpy()
                    for task in logits
                }
                
                all_predictions.append(batch_predictions)
        
        # Convert list of batch predictions into a single dictionary
        predictions = {
            task: np.vstack([batch[task] for batch in all_predictions])
            for task in all_predictions[0]
        }
        
        return predictions

    def objective(trial):
        lr = trial.suggest_float('lr', 1e-5, 2e-3, log=True)
        batch_size = trial.suggest_categorical('batch_size', [192])
        num_epochs = trial.suggest_int('num_epochs', 5, 15)

        model_name = BASE_MODEL


        _, _, val_loss = train_multi_task_model(train_df, val_df, model_name, lr, batch_size, num_epochs)
        
        return val_loss
 


    # create optuna study and optimize hyperparameters
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=10)

    print("Best Hyperparameters:", study.best_params)

    # reload our data to create our best model
    train_df = train_df[['text'] + TRAINING_TASK_COLUMNS].drop_duplicates()
    val_df = val_df[['text'] + VALIDATION_TASK_COLUMNS].drop_duplicates()

    # train model with best hyperparameters
    best_model, best_tokenizer, _ = train_multi_task_model(
        train_df, val_df, BASE_MODEL,
        lr=study.best_params['lr'],
        batch_size=study.best_params['batch_size'],
        num_epochs=study.best_params['num_epochs']
    )

    # Save best model and tokenizer


    torch.save(best_model.state_dict(), MODEL_SAVE_PATH + "_best_model.pth")
    best_tokenizer.save_pretrained(MODEL_SAVE_PATH + '_Tokenizer')

    print("Final model saved to", MODEL_SAVE_PATH + "_best_model.pth")
    print("Final tokenizer saved to", MODEL_SAVE_PATH + '_Tokenizer')

    # Make predictions
    predictions = predict(best_model, best_tokenizer, test_df[['text']])

    # Save predictions
    pred_df = pd.DataFrame({
        'instance_id': test_df['instance_id'],
        'text': test_df['text']
    })

    # add predictions to dataframe
    for task in predictions:
        pred_df[TRAINING_TASK_COLUMNS[int(task)]] = [list(distribution) for distribution in predictions[task]]

    pred_df.to_csv(PREDS_PATH, index=False)

    print("Predictions saved to", PREDS_PATH)

