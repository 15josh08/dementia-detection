import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import pandas as pd

# Load the dataset
df = pd.read_csv('oasis_longitudinal.csv')

# Filter for the first visit data
df = df.loc[df['Visit'] == 1]
df = df.reset_index(drop=True)

# Convert categorical variables to numerical with explicit downcasting
df['M/F'] = df['M/F'].replace({'F': 0, 'M': 1}).infer_objects(copy=False)
df['Group'] = df['Group'].replace({'Converted': 'Demented'}).replace({'Demented': 1, 'Nondemented': 0}).infer_objects(copy=False)

# Drop unnecessary columns
df.drop(['MRI ID', 'Visit', 'Hand'], axis=1, inplace=True)

# Handle missing values
df['SES'].fillna(df.groupby('EDUC')['SES'].transform('median'), inplace=True)
df.dropna(inplace=True)

# Prepare features and target
Y = df['Group'].values
X = df[['M/F', 'Age', 'EDUC', 'SES', 'MMSE', 'eTIV', 'nWBV', 'ASF']]

# Split the data into training and test sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=0)

# Feature scaling
scaler = MinMaxScaler().fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the RandomForestClassifier with best hyperparameters
best_M = 2
best_d = 5
best_m = 7
rf_model = RandomForestClassifier(n_estimators=best_M, max_features=best_d,
                                   max_depth=best_m, random_state=0)
rf_model.fit(X_train_scaled, Y_train)

# Save the trained model to a pickle file
with open('random_forest_model.pkl', 'wb') as f:
    pickle.dump(rf_model, f)

# Save the scaler object to a pickle file
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
