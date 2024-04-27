from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)


model = joblib.load("random_forest_classifier_model(13).pkl")
scaler = joblib.load('scaler5.pkl')

label_map = {
    'Backdoor': 'Backdoor',
    'DoS': 'DoS',
    'Exploits': 'Exploits',
    'Fuzzers': 'Fuzzers',
    'Reconnaissance': 'Reconnaissance',
    'Shellcode': 'Shellcode Injection',
    'Worms': 'Worms'
}

events = {
    'Online Course Registration': {'ignored_attack': 'DoS'},
    'Networking Class Scanning': {'ignored_attack': 'Reconnaissance'},
    'Laboratory Software Updates': {'ignored_attack': 'Exploits'},
    'Remote Collaboration Tools': {'ignored_attack': 'Backdoor'},
    'Cybersecurity Penetration Testing': {'ignored_attack': 'Generic'},
    'Software Testing Projects': {'ignored_attack': 'Fuzzers'},
    'Data Mining Research': {'ignored_attack': 'Analysis'},
    'Exploit Development Exercises': {'ignored_attack': 'Shellcode'},
    'Distributed Computing Projects': {'ignored_attack': 'Worms'},
    'No Event': {'ignored_attack': None}  
}

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    selected_event = 'No Event'  
    
    if request.method == 'POST':
        #check if a file was submitted
        if 'file' not in request.files:
            return render_template('index.html', error='No file part')
        
        file = request.files['file']
        
        #check if file is empty
        if file.filename == '':
            return render_template('index.html', error='No selected file')
        
        #check if the file is of the correct format
        if file:
           
            df = pd.read_csv(file, delimiter="\t")
            
            columns_to_drop = ['id', 'label', 'attack_cat', 'sloss', 'dloss', 'dwin', 'ct_ftp_cmd']
            df = df.drop(columns=[col for col in columns_to_drop if col in df.columns], axis=1)
            
            
            label_encoder = LabelEncoder()
            categorical_columns = ['proto', 'service', 'state']   

            #apply label encoding to categorical columns
            for column in categorical_columns:
                if column in df.columns:
                    df[column] = label_encoder.fit_transform(df[column])
            
            
            scaled_values = scaler.transform(df)
            
            
            predictions = model.predict(scaled_values)
            predicted_attacks = [label_map.get(prediction, 'Unknown') for prediction in predictions]
            
            
            selected_event = request.form.get('event')
            
            
            probabilities = model.predict_proba(scaled_values)
            
            
            ignored_attack = events[selected_event]['ignored_attack']
            
            
            attack_prob_sum = {}
            for probs in probabilities:
                for i, prob in enumerate(probs * 100):
                    attack_prob_sum[model.classes_[i]] = attack_prob_sum.get(model.classes_[i], 0) + prob
            
            
            num_rows = len(df)
            mean_attack_probabilities = {attack: prob_sum / num_rows for attack, prob_sum in attack_prob_sum.items()}
            
            #always add 'No Attack' class and its probability
            mean_attack_probabilities['No Attack'] = mean_attack_probabilities.get('No Attack', 0)
            mean_attack_probabilities['No Attack'] += mean_attack_probabilities.get('Normal', 0)
            mean_attack_probabilities['No Attack'] += mean_attack_probabilities.get('Generic', 0)
            mean_attack_probabilities['No Attack'] += mean_attack_probabilities.get('Analysis', 0)
            
            #if the selected event is 'No Event', do not ignore anything
            if selected_event != 'No Event':
                if ignored_attack and ignored_attack != 'No Attack':
                    ignored_attack_prob = mean_attack_probabilities.get(ignored_attack, 0)
                    mean_attack_probabilities['No Attack'] += ignored_attack_prob
            
            mean_attack_probabilities.pop('Normal', None)
            mean_attack_probabilities.pop('Generic', None)
            mean_attack_probabilities.pop('Analysis', None)
            mean_attack_probabilities.pop(ignored_attack, None)
            
            #precompute the attack with the maximum probability
            max_probability_attack = max(mean_attack_probabilities, key=mean_attack_probabilities.get)
            
            return render_template('index.html', probabilities=mean_attack_probabilities, selected_event=selected_event)
    
    return render_template('index.html', selected_event=selected_event)


if __name__ == '__main__':
    app.run(debug=True)
