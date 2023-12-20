import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

#Load the data
train_data = pd.read_csv('/kaggle/input/csse4375-assignment1/4375train.csv')
test_data = pd.read_csv('/kaggle/input/csse4375-assignment1/4375test.csv')

test_id = test_data['id']
X_train = train_data['sentence']
y_train = train_data['label']
X_test = test_data['sentence']

# Split the data into training and validation sets
X_train_data, X_val_data, y_train_data, y_val_data = train_test_split(X_train, y_train)

# Vectorize the sentences
vectorizer = TfidfVectorizer(max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train_data)
X_val_vec = vectorizer.transform(X_val_data)

#Train the model
model = RandomForestClassifier()
model.fit(X_train_vec, y_train_data)

# Make predictions on the validation set
y_pred = model.predict(X_val_vec)

#Function that can be run to test sentences on traine input data
def test_sentence(input_sentence):
    input_sentence_vec = vectorizer.transform([input_sentence])
    prediction = model.predict(input_sentence_vec)
    return prediction

prediction = ""
while True:
    input_sentence = input("Enter a sentence to test: ")
    if input_sentence.lower() == 'end':
        break
    prediction = test_sentence(input_sentence)
    print("Prediction:", prediction[0])
  
#Accuracy calculation
accuracy = accuracy_score(y_val_data, y_pred)
print("Accuracy of Model = ", accuracy * 100, "%")

# Make predictions on the test set
X_test_vec = vectorizer.transform(X_test)
y_test_pred = model.predict(X_test_vec)

# Save the submission file to the output folder
submission = pd.DataFrame({'id': test_id, 'label': y_test_pred})
submission.to_csv('/kaggle/working/submission.csv', index=False)
