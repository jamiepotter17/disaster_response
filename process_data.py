import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):

    # read in both csv files
    messages = pd.read_csv(messages_filepath, index_col='id')
    categories = pd.read_csv(categories_filepath, index_col='id')

    # merge dataframes
    df = pd.merge(messages, categories, on='id', how='outer')
    return df

def clean_data(df):

    # create a dataframe of the 36 individual category columns
    categories = df.categories.str.split(';', expand=True)

    # rename column names
    row = column_names = list(categories.iloc[0,:].str[:-2])
    categories.columns = column_names

    for column in categories:
    # set each value to be the last character of the string and convert column
    # from string to numeric
        categories[column] = categories[column].str[-1:].astype(int)

    # Replace the '2' values in the 'related' column with '1's:
    df['related']=df['related'].replace(2, 1)
    
    # drop the original categories column from `df`
    df.drop(columns='categories', inplace=True)

    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis=1)

    # drop duplicates
    df = df.drop_duplicates()

    return df

def save_data(df, database_filename):
    engine = create_engine('sqlite:///./data/' + database_filename)
    df.to_sql('messages', engine, if_exists='replace', index=False)
    pass


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)

        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)

        print('Cleaned data saved to database!')

    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
