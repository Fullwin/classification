import pandas as pd
stroke = pd.read_csv('stroke_cleaned.csv')

df = stroke.copy()

encode = ['gender','ever_married','work_type','Residence_type','smoking_status']

#Converting categorical to numerical 
for col in encode:
    dummy = pd.get_dummies(df[col], prefix=col) #Convert categorical varibales to dummy numbers
    df = pd.concat([df,dummy], axis=1) #Add them to the dataframe
    del df[col] #Delete old categorical columns

X = df.drop('stroke', axis=1)
Y = df['stroke']

# Random Forest Model 
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier()
clf.fit(X, Y)

# Save Model 
import pickle
pickle.dump(clf, open('stroke_clf.pkl', 'wb'))