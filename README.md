# PREDICTING-STAR-PROPERTIES
## Abstract:
This project explores the application of predictive modeling techniques to analyze stellar properties, with an emphasis on extending the scope to include Star Age Estimation and Exoplanet Potential evaluation. The study integrates advanced data science methodologies, leveraging classification and regression algorithms to predict spectral types, assess stellar collapse fates, and estimate stellar ages. It also incorporates exoplanetary potential indicators to evaluate the likelihood of planetary systems associated with stars.
### Code
```py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

# Load the dataset
df = pd.read_excel("/content/updated_star_data_with_age.xlsx")

# Drop rows with missing values
df = df.dropna()

# Encode categorical variables
label_encoder_spectral = LabelEncoder()
df['Star Spectral Type'] = label_encoder_spectral.fit_transform(df['Star Spectral Type'])
spectral_classes = label_encoder_spectral.classes_  # Save the spectral types for inverse mapping

label_encoder_fate = LabelEncoder()
df['Evolutionary Fate'] = label_encoder_fate.fit_transform(df['Evolutionary Fate'])
fate_classes = label_encoder_fate.classes_  # Save the evolutionary fate classes for inverse mapping

# Define features and targets for different tasks
features = ['Temperature (K)', 'Radius (Solar Radii)', 'Metallicity [Fe/H]', 'Luminosity (Solar Units)', 'Mass (Solar Masses)']

# Task-specific targets
target_spectral_type = 'Star Spectral Type'
target_age = 'Age (Gyr)'  # Assuming 'Age' column exists
target_evolutionary_fate = 'Evolutionary Fate'
target_exoplanet = 'Exoplanet Host Prediction'

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[features])

# Train-test split for different tasks
X_train, X_test, y_train_class, y_test_class = train_test_split(X_scaled, df[target_spectral_type], test_size=0.2, random_state=42)
X_train_age, X_test_age, y_train_age, y_test_age = train_test_split(X_scaled, df[target_age], test_size=0.2, random_state=42)
X_train_fate, X_test_fate, y_train_fate, y_test_fate = train_test_split(X_scaled, df[target_evolutionary_fate], test_size=0.2, random_state=42)
X_train_exoplanet, X_test_exoplanet, y_train_exoplanet, y_test_exoplanet = train_test_split(X_scaled, df[target_exoplanet], test_size=0.2, random_state=42)
```
# --- Neural Network for Spectral Type Classification ---
model_class = Sequential([
    Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dense(len(np.unique(y_train_class)), activation='softmax')
])

model_class.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model_class.fit(X_train, y_train_class, epochs=50, batch_size=16, verbose=1, validation_split=0.2)

# --- Neural Network for Stellar Age Prediction ---
model_age = Sequential([
    Dense(128, activation='relu', input_shape=(X_train_age.shape[1],)),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dense(1)  # Regression output layer
])

model_age.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
model_age.fit(X_train_age, y_train_age, epochs=50, batch_size=16, verbose=1, validation_split=0.2)

# --- Neural Network for Evolutionary Fate Classification ---
model_fate = Sequential([
    Dense(128, activation='relu', input_shape=(X_train_fate.shape[1],)),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dense(len(np.unique(y_train_fate)), activation='softmax')
])

model_fate.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model_fate.fit(X_train_fate, y_train_fate, epochs=50, batch_size=16, verbose=1, validation_split=0.2)

# --- Neural Network for Exoplanet Potential Classification ---
model_exoplanet = Sequential([
    Dense(128, activation='relu', input_shape=(X_train_exoplanet.shape[1],)),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid')  # Binary classification output layer
])

model_exoplanet.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model_exoplanet.fit(X_train_exoplanet, y_train_exoplanet, epochs=50, batch_size=16, verbose=1, validation_split=0.2)

# --- Input for predicting new star data ---
def predict_new_star_data():
    # Prompt user for input data dynamically
    temperature = float(input("Enter the star's Temperature (K): "))
    radius = float(input("Enter the star's Radius (in Solar Radii): "))
    metallicity = float(input("Enter the star's Metallicity [Fe/H]: "))
    luminosity = float(input("Enter the star's Luminosity (in Solar Units): "))
    mass = float(input("Enter the star's Mass (in Solar Masses): "))

    # Prepare new data for prediction
    new_data = [temperature, radius, metallicity, luminosity, mass]

    # Standardize the input data using the same scaler
    new_data_scaled = scaler.transform([new_data])

    # Spectral Type Prediction
    spectral_type_prob = model_class.predict(new_data_scaled)
    spectral_type_pred = np.argmax(spectral_type_prob)  # Get the predicted class index
    spectral_type = spectral_classes[spectral_type_pred]  # Map class index back to label

    # Stellar Age Prediction
    predicted_age = model_age.predict(new_data_scaled)
    predicted_age_value = predicted_age.item()  # Access the scalar value of age

    # Evolutionary Fate Prediction
    fate_prob = model_fate.predict(new_data_scaled)
    fate_pred = np.argmax(fate_prob)  # Get the predicted class index for evolutionary fate
    evolutionary_fate = fate_classes[fate_pred]  # Map class index back to label

    # Exoplanet Potential Prediction
    exoplanet_prob = model_exoplanet.predict(new_data_scaled)
    exoplanet_potential = (exoplanet_prob > 0.5).astype("int32")  # Convert to binary classification

    # Output predictions
    print(f"\nPredicted Spectral Type: {spectral_type}")  # 'G', 'F', etc.
    print(f"Predicted Stellar Age: {predicted_age_value:.2f} Gyr")  # Age in billion years
    print(f"Predicted Evolutionary Fate: {evolutionary_fate}")  # 'White Dwarf', 'Neutron Star', etc.
    print(f"Exoplanet Potential: {'Yes' if exoplanet_potential == 1 else 'No'}")

# Run the function to predict new star data
predict_new_star_data()

```
### Output
![image](https://github.com/user-attachments/assets/9d061f09-a267-4a92-8ac1-6d5705f5b422)



