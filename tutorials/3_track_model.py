import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import mlflow
import mlflow.keras
import numpy as np

from mlflow.models import infer_signature

# Define a simple Keras model
def create_model():
    model = keras.Sequential([
        layers.Dense(32, activation='relu', input_shape=(10,)),
        layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

# Generate dummy data
X = np.random.randn(100, 10)
y = np.random.randn(100, 1)

def train():
    model = create_model()
    
    mlflow.set_experiment("simple_keras_experiment")
    
    with mlflow.start_run():
        mlflow.log_param("learning_rate", 0.01)
        mlflow.log_param("batch_size", 10)
        
        history = model.fit(X, y, epochs=10, batch_size=10, verbose=1)
        
        for epoch, loss in enumerate(history.history['loss']):
            mlflow.log_metric("loss", loss, step=epoch)
            print(f"Epoch {epoch+1}, Loss: {loss}")

        # Log the trained Keras model with a signature
        mlflow.keras.log_model(model, "model", signature=infer_signature(np.random.randn(1, 10), np.random.randn(1, 1))) 

if __name__ == "__main__":
    train()
