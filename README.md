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

2. Run the web app: `python run.py`

3. Copy the URL into your browser - e.g. http://192.168.1.117:3000/ from
the line 'Running on http://192.168.1.117:3000/ (Press CTRL+C to quit)'.

Please note also that the model used in the classifier has been fine-tuned on
the curated dataset supplied by Figure Eight, and the pipeline will use those
hyperparameters. In order to carry out a new Grid Search (or, or accurately,
a Randomised Search), you will need to run 'ML Pipeline Preparation.ipynb' to
obtain the best hyperparameters, and then update the model manually in
'train_classifier.py'.

### File list:

* .gitignore - tells GitHub which files it doesn't need to monitor.
* 'ETL Pipeline Preparation.ipynb' - Jupyter Notebook file used to develop the
code for process_data.py.
* 'ML Pipeline Preparation.ipynb' - Jupyter Notebook file used to develop the
code for train_classifier.py.
* README.MD - this file.
* graphs.py - python module that contains get_graphs(), a function that returns
the graphs that ultimately get displayed on master.html.
* process_data.py - python script that runs an ETL pipeline, loading data into
an SQLite database, specifically a table titled 'messages'.
* requirements.txt - things should work fine in a standard Anaconda
environment, but here is a list of all the packages used in my Anaconda
environment in case you have compatibility issues.  Use
'pip install requirements.txt' to install the packages.
* run.py - python script that runs the Flash app that routes traffic to the
right places.
* train_classifier.py - python script that runs a ML pipeline, transforming
data from an SQLite database into a pickle file. 
