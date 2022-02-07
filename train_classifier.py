import sys
import numpy as np
import pandas as pd
from sqlalchemy import create_engine
import re
import joblib
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.multioutput import MultiOutputClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

def load_data(database_filepath):
    engine = create_engine('sqlite:///'+ database_filepath)
    df = pd.read_sql_table('messages', engine)
    X = df['message']
    category_names = list(df.columns[3:])
    Y = df[category_names]
    return X, Y, category_names

def tokenize(text):
    '''
    INPUT - (list) list of strings to be tokenised.
    OUTPUT - (list) list of tokenised strings.
    '''

    # Replace URLs with string 'urlsupplied'
    urlregex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    urlsupplied_text = re.sub(urlregex, 'urlsupplied', text)

    # remove punctuation and convert to small letters
    small_unpunc_text = re.sub(r"[^A-Za-z0-9']", ' ', urlsupplied_text).lower()

    # tokenize text
    tokenized_text = word_tokenize(small_unpunc_text)

    # lemmatize test
    lemmatizer = WordNetLemmatizer()
    lemmatized_text = []
    for item in tokenized_text:
        lemmatized_text.append(lemmatizer.lemmatize(item))

    return lemmatized_text

def build_model():
    '''
    INPUT -
    OUTPUT pipeline (Pipeline object) returns pipeline object with the
    parameters found from a prior use of a randomised search.
    '''
    pipeline = Pipeline([
        ("vect", CountVectorizer(tokenizer=tokenize,
        lowercase=False, max_df=0.2, ngram_range=(1, 2))),
        ("tfidf", TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier(
        n_estimators=100, min_samples_split=8, min_samples_leaf=1)))
        ])

    return pipeline

def evaluate_model(fittedmodel, X_test, Y_test, category_names):
    '''
    Iterates over columns of response variables. Prints macro and weighted
    average f1_Score, recall and precision.
    INPUTS:
    fittedmodel - sci-kit learn classifier object that should already have been
    fitted with the data.
    X_test (1d numpy array, pandas series) - explanatory variables you wish to
    test the model with.
    Y_test (2d numpy array, pandas df) - reponse variables you wish to test the
    model with.
    category_names (list) - list of category_names from the dataframe.
    OUTPUTS:
    -
    '''
    predictions = fittedmodel.predict(X_test)

    f1_scores_m = []
    precision_scores_m = []
    recall_scores_m = []
    f1_scores_w = []
    precision_scores_w = []
    recall_scores_w = []

    for i in range(Y_test.shape[1]):

        score_dict = classification_report(predictions[:][i], np.array(Y_test)[:][i],
                                                      zero_division=0, output_dict=True)

        f1_scores_m.append(score_dict['macro avg']['f1-score'])
        precision_scores_m.append(score_dict['macro avg']['precision'])
        recall_scores_m.append(score_dict['macro avg']['recall'])
        f1_scores_w.append(score_dict['weighted avg']['f1-score'])
        precision_scores_w.append(score_dict['weighted avg']['precision'])
        recall_scores_w.append(score_dict['weighted avg']['recall'])

        print("Scores in category \'{}\':".format(category_names[i]))
        print("    f1 score:  {:.2f}.".format(f1_scores_m[-1]))
        print("    Precision: {:.2f}.".format(precision_scores_m[-1]))
        print("    Recall:    {:.2f}.".format(recall_scores_m[-1]))

    print("\nAverage macro f1 score is {:.2f}.".format(np.array(f1_scores_m).mean()))
    print("Average macro precision score is {:.2f}.".format(np.array(precision_scores_m).mean()))
    print("Average macro recall score is {:.2f}.".format(np.array(recall_scores_m).mean()))
    print("Average weighted f1 score is {:.2f}.".format(np.array(f1_scores_w).mean()))
    print("Average weighted precision score is {:.2f}.".format(np.array(precision_scores_w).mean()))
    print("Average weighted recall score is {:.2f}.".format(np.array(recall_scores_w).mean()))

def save_model(model, model_filepath):
    with open('./models/' + model_filepath, "wb") as f:
        joblib.dump(model, f, compress='zlib')

def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))

        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

        print('Building model...')
        model = build_model()

        print('Training model...')
        model.fit(X_train, Y_train)

        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
