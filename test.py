import pickle
import pandas as pd

model = pickle.load(open('heart/lr.pkl', 'rb'))

sample = pd.DataFrame({
    'Age': [52],
    'Sex': [1],
    'ChestPainType': [2],
    'RestingBP': [130],
    'Cholesterol': [200],
    'FastingBS': [1],
    'RestingECG': [0],
    'MaxHR': [150],
    'ExerciseAngina': [0],
    'Oldpeak': [2.3],
    'ST_Slope': [1]
})

print(model.predict(sample))

