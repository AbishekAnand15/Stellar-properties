# PREDICTING-STAR-PROPERTIES
## Abstract:
This project explores the application of predictive modeling techniques to analyze stellar properties, with an emphasis on extending the scope to include Star Age Estimation and Exoplanet Potential evaluation. The study integrates advanced data science methodologies, leveraging classification and regression algorithms to predict spectral types, assess stellar collapse fates, and estimate stellar ages. It also incorporates exoplanetary potential indicators to evaluate the likelihood of planetary systems associated with stars.
### Code
```py
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, confusion_matrix

def classify_spectral_type(temp):
    """Classify spectral type based on temperature."""
    if temp > 30000:
        return "O"
    elif 10000 <= temp <= 30000:
        return "B"
    elif 7500 <= temp < 10000:
        return "A"
    elif 6000 <= temp < 7500:
        return "F"
    elif 5200 <= temp < 6000:
        return "G"
    elif 3700 <= temp < 5200:
        return "K"
    else:
        return "M"

def predict_collapse_fate(mass):
    """Predict the collapse fate of a star based on mass."""
    if mass < 1.44:
        return "White Dwarf"
    elif 1.44 <= mass <= 2.5:
        return "Neutron Star"
    else:
        return "Black Hole"

def estimate_age(mass):
    """Estimate the age of the star based on its mass."""
    if mass <= 0.5:
        return 13.8
    elif 0.5 < mass <= 1.0:
        return 10
    elif 1.0 < mass <= 3.0:
        return 1
    else:
        return 0.1

def exoplanet_potential(luminosity):
    """Estimate the exoplanet potential based on luminosity."""
    habitable_zone_min = (luminosity ** 0.5) * 0.95
    habitable_zone_max = (luminosity ** 0.5) * 1.37
    
    if habitable_zone_min < 0.1:
        return "No Potential"
    elif habitable_zone_max > 1.5:
        return "Possible Habitable Zone"
    else:
        return "Likely Habitable Zone"

def simulate_data():
    """Simulate some data for testing (true labels and predictions)."""
    # Simulated true values and predicted values for spectral type and collapse fate
    true_spectral_types = ['O', 'B', 'A', 'F', 'G', 'K', 'M']
    predicted_spectral_types = ['O', 'B', 'A', 'F', 'G', 'K', 'K']  # Simulate slight errors

    true_fates = ['White Dwarf', 'Neutron Star', 'Black Hole', 'White Dwarf', 'Neutron Star']
    predicted_fates = ['White Dwarf', 'Neutron Star', 'Black Hole', 'White Dwarf', 'Black Hole']  # Simulate errors

    return true_spectral_types, predicted_spectral_types, true_fates, predicted_fates

def calculate_metrics(true_values, predicted_values):
    """Calculate accuracy, precision, and confusion matrix."""
    accuracy = accuracy_score(true_values, predicted_values)
    precision = precision_score(true_values, predicted_values, average='weighted', zero_division=0)
    conf_matrix = confusion_matrix(true_values, predicted_values)

    return accuracy, precision, conf_matrix

# User Input Section
print("Enter the stellar parameters for a star:")

# Gather inputs for the star
mass = float(input("Mass (in solar masses): "))
radius = float(input("Radius (in solar radii): "))
temperature = float(input("Temperature (in Kelvin): "))
luminosity = float(input("Luminosity (in solar luminosities): "))
metallicity = float(input("Metallicity (relative to the Sun): "))

# Determine derived attributes
spectral_type = classify_spectral_type(temperature)
collapse_fate = predict_collapse_fate(mass)
age = estimate_age(mass)
exoplanet_potential_status = exoplanet_potential(luminosity)

# Simulate some data for classification metrics
true_spectral_types, predicted_spectral_types, true_fates, predicted_fates = simulate_data()

# Calculate metrics for Spectral Type
accuracy_spectral, precision_spectral, conf_matrix_spectral = calculate_metrics(true_spectral_types, predicted_spectral_types)

# Calculate metrics for Collapse Fate
accuracy_fate, precision_fate, conf_matrix_fate = calculate_metrics(true_fates, predicted_fates)

# Display results
print("\n--- Stellar Properties ---")
print(f"Mass: {mass} Solar Masses")
print(f"Radius: {radius} Solar Radii")
print(f"Temperature: {temperature} K")
print(f"Luminosity: {luminosity} Solar Luminosities")
print(f"Metallicity: {metallicity} (relative to the Sun)")

print("\n--- Derived Attributes ---")
print(f"Spectral Type: {spectral_type}")
print(f"Collapse Fate: {collapse_fate}")
print(f"Star Age Estimate: {age} Billion Years")
print(f"Exoplanet Potential: {exoplanet_potential_status}")

print("\n--- Classification Metrics ---")
print(f"Spectral Type Accuracy: {accuracy_spectral:.2f}")
print(f"Spectral Type Precision: {precision_spectral:.2f}")
print(f"Spectral Type Confusion Matrix:\n{conf_matrix_spectral}")
print(f"Collapse Fate Accuracy: {accuracy_fate:.2f}")
print(f"Collapse Fate Precision: {precision_fate:.2f}")
print(f"Collapse Fate Confusion Matrix:\n{conf_matrix_fate}")

```
### Output
![Screenshot 2024-11-16 183755](https://github.com/user-attachments/assets/ab6ea003-9677-418d-b5b6-72a577598369)


