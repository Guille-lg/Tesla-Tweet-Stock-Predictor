import pandas as pd
import os
from dotenv import load_dotenv
import openai

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

system_prompt = """You are a data scientist labelling tweets. Given a tweet, classify it into one of the following categories:

Tesla Products
Customer Experience
Performance & Innovation
Financial News
Environmental Impact
Industry News
Charging Infrastructure

If the tweet does not clearly fit into any of these categories or is not relevant, classify it as "Not relevant".

Example:

Tweet:

Tesla's new Model S Plaid is the fastest production car ever made!

Category:

Tesla Products

Example:

Tweet:

I'm having a terrible time with Tesla customer service. They've been ignoring my emails for weeks.

Category:

Customer Experience

I want you to be very rigorous with the labelling. You have to respond with the following phrase: Category: [response]"""

df = pd.read_csv(filepath_or_buffer='C:/Users/34644/Desktop/Cursos/Curso_CEI/Tesla-Tweet-Stock-Predictor/data/clean/tesla_tweets_unlabelled.csv')
test_df = df.sample(100)


df_labelled = pd.DataFrame(columns=['Link', 'Date', 'Text', 'Category'])
progress = 0

for index, row in test_df.iterrows():
    try:
        completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": row['Text']}
        ],
            max_tokens=10,
            temperature=0.9,
        )
        answer = completion.choices[0].message['content'].replace('Category: ', '')
        df_labelled.loc[len(df_labelled.index)] = [row['Link'], row['Date'], row['Text'], answer]
        progress += 1
        if progress % 10 == 0:
            print(f"Progress: {progress}/{len(test_df.index)}")
    except Exception as e:
        print(e)
        continue

df_aux = pd.read_csv('C:/Users/34644/Desktop/Cursos/Curso_CEI/Tesla-Tweet-Stock-Predictor/data/clean/tesla_tweets_unlabelled.csv')

#comprobar que ninguna fila coincide con la que ya esta en el df_aux

mask = df_labelled['Link'].isin(df_aux['Link'])
df_labelled = df_labelled[~mask]

pd.concat([df_aux, df_labelled]).to_csv('C:/Users/34644/Desktop/Cursos/Curso_CEI/Tesla-Tweet-Stock-Predictor/data/clean/tesla_tweets_unlabelled.csv', index=False)