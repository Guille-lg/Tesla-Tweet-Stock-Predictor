from datasets import load_dataset
from sentence_transformers.losses import CosineSimilarityLoss
from setfit import SetFitModel, SetFitTrainer, sample_dataset
from sklearn.model_selection import train_test_split
import pandas as pd
import os
from dotenv import load_dotenv

load_dotenv()

labelled_route = os.getenv('TWEET_CATEGORY_LABELLED_DATA_PATH')
train_filepath = os.getenv('TWEET_CATEGORY_LABELLED_TRAIN_DATA_PATH')
test_filepath = os.getenv('TWEET_CATEGORY_LABELLED_TEST_DATA_PATH')
model_filepath = os.getenv('TWEET_CATEGORY_MODEL_PATH')
model_name = 'sentence-transformers/all-mpnet-base-v2'

label_map = {
    "Tesla Products": 0,
    "Customer Experience": 1,
    "Performance & Innovation": 2,
    "Financial News": 3,
    "Environmental Impact": 4,
    "Industry News": 5,
    "Charging Infrastructure": 6,
    'not relevant': 7
}

print('Loading data...')

df = pd.read_csv(labelled_route)
df.drop(['Link', 'Date'], axis=1, inplace=True)

df['Category'] = df['Category'].str.lower().str.strip('.')

train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

train_df.rename(columns={'Text': 'text', 'Category' : 'label'}, inplace=True)
test_df.rename(columns={'Text': 'text', 'Category' : 'label'}, inplace=True)


train_df['label'] = train_df['label'].map(label_map)
test_df['label'] = test_df['label'].map(label_map)

print(train_df['label'].value_counts())

train_df.to_csv(train_filepath, index=False)
test_df.to_csv(test_filepath, index=False)

dataset = load_dataset(
    'csv', 
    data_files={
        'train': train_filepath,
        'test': test_filepath
    }
)


print('Data loaded.')

print('Loading model...')

model = SetFitModel.from_pretrained(model_name)

print('Model loaded.')

print('Training...')

trainer = SetFitTrainer(
    model=model,
    train_dataset=dataset['train'],
    eval_dataset=dataset['test'],
    loss_class=CosineSimilarityLoss,
    batch_size=16,
    num_iterations=50, 
    num_epochs=1
)

trainer.train()

print('Training finished.')

print('Saving model...')

trainer.model.save_pretrained(save_directory=model_filepath)

print('Evaluating...')
metrics = trainer.evaluate()
print(metrics)


