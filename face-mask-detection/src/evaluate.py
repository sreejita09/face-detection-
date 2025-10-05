def evaluate_model(model, test_data, test_labels):
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

    predictions = model.predict(test_data)
    predicted_classes = (predictions > 0.5).astype("int32")

    accuracy = accuracy_score(test_labels, predicted_classes)
    precision = precision_score(test_labels, predicted_classes)
    recall = recall_score(test_labels, predicted_classes)
    f1 = f1_score(test_labels, predicted_classes)

    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1
    }

    return metrics

def load_test_data(test_data_path):
    import os
    import cv2
    import numpy as np

    images = []
    labels = []

    for filename in os.listdir(test_data_path):
        img_path = os.path.join(test_data_path, filename)
        image = cv2.imread(img_path)
        image = cv2.resize(image, (224, 224))  # Resize to match model input
        images.append(image)

        # Assuming labels are encoded in the filename (e.g., mask_1.jpg, no_mask_0.jpg)
        label = 1 if 'mask' in filename else 0
        labels.append(label)

    return np.array(images), np.array(labels)

if __name__ == "__main__":
    import sys
    from model import load_model  # Assuming a function to load the trained model

    model_path = sys.argv[1]
    test_data_path = sys.argv[2]

    model = load_model(model_path)
    test_data, test_labels = load_test_data(test_data_path)
    metrics = evaluate_model(model, test_data, test_labels)

    print("Evaluation Metrics:")
    for key, value in metrics.items():
        print(f"{key}: {value}")