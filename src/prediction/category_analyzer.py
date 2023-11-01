from datasets import load_dataset
from sentence_transformers.losses import CosineSimilarityLoss
from setfit import SetFitModel, SetFitTrainer, sample_dataset
from sklearn.model_selection import train_test_split
import pandas as pd


print('Loading data...')

df = pd.read_csv('C:/Users/34644/Desktop/Cursos/Curso_CEI/Tesla-Tweet-Stock-Predictor/data/clean/tesla_tweets_labelled.csv')
df.drop(['Link', 'Date'], axis=1, inplace=True)

df['Category'] = df['Category'].str.lower().str.strip('.')

train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

train_df.rename(columns={'Text': 'text', 'Category' : 'label'}, inplace=True)
test_df.rename(columns={'Text': 'text', 'Category' : 'label'}, inplace=True)

#map labels to integers
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

train_df['label'] = train_df['label'].map(label_map)
test_df['label'] = test_df['label'].map(label_map)

print(train_df['label'].value_counts())

train_df.to_csv('C:/Users/34644/Desktop/Cursos/Curso_CEI/Tesla-Tweet-Stock-Predictor/data/clean/tesla_tweets_labelled_train.csv', index=False)
test_df.to_csv('C:/Users/34644/Desktop/Cursos/Curso_CEI/Tesla-Tweet-Stock-Predictor/data/clean/tesla_tweets_labelled_test.csv', index=False)

dataset = load_dataset(
    'csv', 
    data_files={
        'train': 'C:/Users/34644/Desktop/Cursos/Curso_CEI/Tesla-Tweet-Stock-Predictor/data/clean/tesla_tweets_labelled_train.csv',
        'test': 'C:/Users/34644/Desktop/Cursos/Curso_CEI/Tesla-Tweet-Stock-Predictor/data/clean/tesla_tweets_labelled_test.csv'
    
    }
)


print('Data loaded.')

print('Loading model...')

model = SetFitModel.from_pretrained(
    "sentence-transformers/all-mpnet-base-v2"
)

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

trainer.model.save_pretrained(save_directory='C:/Users/34644/Desktop/Cursos/Curso_CEI/Tesla-Tweet-Stock-Predictor/src/prediction/output_models/')

print('Evaluating...')
metrics = trainer.evaluate()
print(metrics)


