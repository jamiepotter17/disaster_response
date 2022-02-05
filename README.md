# Disaster Response Project

This is a project for Udacity's Data Scientist Nanodegree
It was commenced on 2012-02-02. The brief was to:

1. Create an ETL pipeline to prepare data curated by Figure Eight Technologies,
a now defunct analytics firm, and load into an SQLite database. These data are
the raw data of messages sent during natural disasters.
2. Develop a machine learning pipeline that categories the data for better
disaster response efforts, and exports the final model as a pickle file.
3. Produce data visualisations using a Flask web app and Plotly.

### Instructions:
1. Run the following commands in the project's root directory to set up your
database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python ./process_data.py data/disaster_messages.csv
        data/disaster_categories.csv data/Disaster_Response.db`
    - To run ML pipeline that trains classifier and saves
        `python ./train_classifier.py data/Disaster_Response.db
        models/classifier.pkl`

2. Go to `app` directory: `cd app`

3. Run your web app: `python run.py`

4. Click the `PREVIEW` button to open the homepage
