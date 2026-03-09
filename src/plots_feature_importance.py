import joblib
import pandas as pd
import matplotlib.pyplot as plt

model = joblib.load("../models/random_forest_log.pkl")

rf = model.named_steps["model"]
features = model.named_steps["preprocess"].get_feature_names_out()

fi = pd.DataFrame({
    "feature": features,
    "importance": rf.feature_importances_
}).sort_values(by="importance", ascending=False).head(15)

plt.figure(figsize=(8,5))
plt.barh(fi["feature"], fi["importance"])
plt.gca().invert_yaxis()
plt.title("Top Feature Importances (Random Forest)")
plt.tight_layout()
plt.show()
