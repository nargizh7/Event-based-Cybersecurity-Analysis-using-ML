### Event-based Cybersecurity Analysis using ML

This repository contains a Python application designed for analyzing cybersecurity threats using machine learning models, specifically focusing on event-based data. The application is developed with Flask and uses a RandomForestClassifier model to predict potential cyber-attacks based on network log data.

#### Repository Structure
- `Training_script.py`: A script that prepares and trains the machine learning model using the UNSW-NB15 dataset. It includes data preprocessing, model training with RandomForestClassifier, and performance evaluation.
- `app.py`: A Flask application that loads the trained model and uses it to predict cyber-attacks. It handles file uploads and displays predictions through a web interface.
- `templates/index.html`: The HTML template for the web application, which includes form handling and result display.

#### Features
- **Data Preprocessing**: Combines training and testing datasets, encodes categorical features, and applies SMOTE for handling class imbalance.
- **Model Training**: Trains a RandomForest model with specified parameters and evaluates its performance using metrics like accuracy, precision, recall, and F1-score.
- **Web Application**: Provides a user interface for uploading event-based network logs, selecting specific types of network events, and viewing predicted cyber-attack probabilities.

#### Usage
1. Run the `Training_script.py` to train the model and save it along with the scaler.
2. Start the Flask application by running `app.py`.
3. Access the web interface, upload a network log file, and select an event to see the predictions.

#### Installation
To set up and run this application, you will need Python and Flask installed on your machine. Clone the repository and install the required Python packages specified in the script comments.

