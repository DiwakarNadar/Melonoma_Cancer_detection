import joblib
from data import data
a=data['Actinic keratoses']
print(a['Name'])
pred_id = joblib.load('pred_id.joblib')
print(pred_id)
