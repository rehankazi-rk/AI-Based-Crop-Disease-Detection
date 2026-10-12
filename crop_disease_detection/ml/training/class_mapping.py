"""Canonical Kaggle PlantVillage labels supported by the deployed classifier."""

SOURCE_TO_CLASS = {
    "Tomato___healthy": "Healthy_Tomato",
    "Tomato___Early_blight": "Early_Blight_Tomato",
    "Tomato___Late_blight": "Late_Blight_Tomato",
    "Tomato___Leaf_Mold": "Leaf_Mold_Tomato",
    "Potato___healthy": "Healthy_Potato",
    "Potato___Early_blight": "Early_Blight_Potato",
    "Potato___Late_blight": "Late_Blight_Potato",
}

# ImageFolder orders class labels lexicographically; the checkpoint uses that same order.
MODEL_CLASS_NAMES = tuple(sorted(SOURCE_TO_CLASS.values()))
