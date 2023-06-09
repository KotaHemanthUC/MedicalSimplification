# Importing stock libraries
import json
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler

# Importing the T5 modules from huggingface/transformers
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


from torch import cuda
device = 'cuda' if cuda.is_available() else 'cpu'
print(device)


# Creating a custom dataset for reading the dataframe and loading it into the dataloader to pass it to the neural network at a later stage for finetuning the model and to prepare it for predictions

class CustomDataset(Dataset):

    def __init__(self, dataframe, tokenizer, source_len, summ_len):
        self.tokenizer = tokenizer
        self.data = dataframe
        self.source_len = source_len
        self.summ_len = summ_len
        self.text = self.data['Simplified Sentence']
        self.ctext = self.data['Original Sentence']

    def __len__(self):
        return len(self.text)

    def __getitem__(self, index):
        ctext = str(self.ctext[index])
        ctext = ' '.join(ctext.split())

        text = str(self.text[index])
        text = ' '.join(text.split())

        source = self.tokenizer.batch_encode_plus(
            [ctext], max_length=self.source_len, pad_to_max_length=True, return_tensors='pt')
        target = self.tokenizer.batch_encode_plus(
            [text], max_length=self.summ_len, pad_to_max_length=True, return_tensors='pt')

        source_ids = source['input_ids'].squeeze()
        source_mask = source['attention_mask'].squeeze()
        target_ids = target['input_ids'].squeeze()
        target_mask = target['attention_mask'].squeeze()

        return {
            'source_ids': source_ids.to(dtype=torch.long),
            'source_mask': source_mask.to(dtype=torch.long),
            'target_ids': target_ids.to(dtype=torch.long),
            'target_ids_y': target_ids.to(dtype=torch.long)
        }

# Creating the training function. This will be called in the main function. It is run depending on the epoch value.
# The model is put into train mode and then we wnumerate over the training loader and passed to the defined network


def train(epoch, tokenizer, model, device, loader, optimizer):
    model.train()
    # print(f"The model that is being run is {model.config.model_type}")
    for _, data in enumerate(loader, 0):
        y = data['target_ids'].to(device, dtype=torch.long)
        y_ids = y[:, :-1].contiguous()
        lm_labels = y[:, 1:].clone().detach()
        lm_labels[y[:, 1:] == tokenizer.pad_token_id] = -100
        ids = data['source_ids'].to(device, dtype=torch.long)
        mask = data['source_mask'].to(device, dtype=torch.long)

        print(f"This is ids {ids.shape}")
        print(f"This is mask {mask.shape}")
        print(f"This is y_ids {y_ids.shape}")
        print(f"This is lm_labels {lm_labels.shape}")

        outputs = model(input_ids=ids, attention_mask=mask,
                        decoder_input_ids=y_ids, labels=lm_labels)
        loss = outputs[0]

        if _ % 500 == 0:
            print(f'Epoch: {epoch}, Loss:  {loss.item()}')

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # xm.optimizer_step(optimizer)
        # xm.mark_step()


def validate(epoch, tokenizer, model, device, loader):
    model.eval()
    predictions = []
    actuals = []
    with torch.no_grad():
        for _, data in enumerate(loader, 0):
            y = data['target_ids'].to(device, dtype=torch.long)
            ids = data['source_ids'].to(device, dtype=torch.long)
            mask = data['source_mask'].to(device, dtype=torch.long)

            generated_ids = model.generate(
                input_ids=ids,
                attention_mask=mask,
                max_length=150,
                num_beams=2,
                repetition_penalty=2.5,
                length_penalty=1.0,
                early_stopping=True
            )
            preds = [tokenizer.decode(
                g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in generated_ids]
            target = [tokenizer.decode(
                t, skip_special_tokens=True, clean_up_tokenization_spaces=True)for t in y]
            if _ % 100 == 0:
                print(f'Completed {_}')

            predictions.extend(preds)
            actuals.extend(target)
    return predictions, actuals


data = ''


def main():
    # WandB – Initialize a new run

    # WandB – Config is a variable that holds and saves hyperparameters and inputs
    # Defining some key variables that will be used later on in the training
    TRAIN_BATCH_SIZE = 2    # input batch size for training (default: 64)
    # input batch size for testing (default: 1000)
    VALID_BATCH_SIZE = 2
    TRAIN_EPOCHS = 1        # number of epochs to train (default: 10)
    VAL_EPOCHS = 1
    LEARNING_RATE = 5e-5    # learning rate (default: 0.01)
    SEED = 42               # random seed (default: 42)
    MAX_LEN = 500
    SUMMARY_LEN = 300

    # Set random seeds and deterministic pytorch for reproducibility
    torch.manual_seed(SEED)  # pytorch random seed
    np.random.seed(SEED)  # numpy random seed
    torch.backends.cudnn.deterministic = True

    # tokenzier for encoding the text
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")


    model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")
    # tokenizer_flan =  #T5Tokenizer.from_pretrained("t5-base")

    # Importing and Pre-Processing the domain data
    # Selecting the needed columns only.
    # Adding the summarzie text in front of the text. This is to format the dataset similar to how T5 model was trained for summarization task.
    '''
    df1 = pd.read_csv('./news_summary.csv',encoding='latin-1')
    df1 = df1[['text','ctext']]
    df1.ctext = 'summarize: ' + df1.ctext
    print(df1.head())
    '''
    df = pd.DataFrame(columns=['Original Sentence', 'Simplified Sentence'])

    with open('data_final_1024.json', 'r') as f:
        # returns JSON object as
        # a dictionary
        data = json.load(f)

    for i in range(len(data)):
        # first is the flan-t5 simplification
        paper = data[i]  # this is a dictionary
        org = paper['abstract']
        simp = paper['pls']
        df.loc[len(df)] = [org, simp]

    df['Original Sentence'] = 'Simplify: ' + df['Original Sentence']
    print(df.head())
    # Creation of Dataset and Dataloader
    # Defining the train size. So 80% of the data will be used for training and the rest will be used for validation.
    train_size = 0.8
    train_dataset = df.sample(frac=train_size, random_state=SEED)
    val_dataset = df.drop(train_dataset.index).reset_index(drop=True)
    train_dataset = train_dataset.reset_index(drop=True)

    print("FULL Dataset: {}".format(df.shape))
    print("TRAIN Dataset: {}".format(train_dataset.shape))
    print("TEST Dataset: {}".format(val_dataset.shape))

    # Creating the Training and Validation dataset for further creation of Dataloader
    training_set = CustomDataset(
        train_dataset, tokenizer, MAX_LEN, SUMMARY_LEN)
    val_set = CustomDataset(val_dataset, tokenizer,
                            MAX_LEN, SUMMARY_LEN)

    # Defining the parameters for creation of dataloaders
    train_params = {
        'batch_size': TRAIN_BATCH_SIZE,
        'shuffle': True,
        'num_workers': 0
    }

    val_params = {
        'batch_size': VALID_BATCH_SIZE,
        'shuffle': False,
        'num_workers': 0
    }

    # Creation of Dataloaders for testing and validation. This will be used down for training and validation stage for the model.
    training_loader = DataLoader(training_set, **train_params)
    val_loader = DataLoader(val_set, **val_params)

    # Defining the model. We are using t5-base model and added a Language model layer on top for generation of Summary.
    # Further this model is sent to device (GPU/TPU) for using the hardware.
    # T5ForConditionalGeneration.from_pretrained("t5-base")
    model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")
    model = model.to(device)

    # Defining the optimizer that will be used to tune the weights of the network in the training session.
    optimizer = torch.optim.Adam(
        params=model.parameters(), lr=LEARNING_RATE)

    # Log metrics with wandb

    # Training loop
    print('Initiating Fine-Tuning for the model on our dataset')

    for epoch in range(TRAIN_EPOCHS):
        train(epoch, tokenizer, model, device, training_loader, optimizer)

    # Validation loop and saving the resulting file with predictions and acutals in a dataframe.
    # Saving the dataframe as predictions.csv
    print('Now generating summaries on our fine tuned model for the validation dataset and saving it in a dataframe')
    for epoch in range(VAL_EPOCHS):
        predictions, actuals = validate(
            epoch, tokenizer, model, device, val_loader)
        final_df = pd.DataFrame(
            {'Generated Text': predictions, 'Actual Text': actuals})
        final_df.to_csv('./predictions_flan.csv')
        print('Output Files generated for review')

    model.save_pretrained("./flan/")
    tokenizer.save_pretrained("./flan/")

    return model


if __name__ == '__main__':
    fine_tuned_model = main()
