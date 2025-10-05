def load_config(config_path):
    import json
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config

def visualize_results(results):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 5))
    plt.plot(results['epochs'], results['accuracy'], label='Accuracy')
    plt.plot(results['epochs'], results['loss'], label='Loss')
    plt.title('Model Performance')
    plt.xlabel('Epochs')
    plt.ylabel('Metrics')
    plt.legend()
    plt.show()

def save_model(model, model_path):
    model.save(model_path)

def load_model(model_path):
    from tensorflow.keras.models import load_model
    return load_model(model_path)