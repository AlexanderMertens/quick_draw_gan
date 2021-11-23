import mlflow

from models.train_model import train_model

mlflow.start_run()
train_model(num_epochs=11, num_batch=16, batch_size=32)
mlflow.end_run()
