import tensorflow as tf

from mlflow.tracking.fluent import end_run, log_artifact, start_run
from mlflow.keras import log_model

from models.train_model import train_model

tf.config.threading.set_intra_op_parallelism_threads(2)
tf.config.threading.set_inter_op_parallelism_threads(2)
start_run()

gan = train_model(num_epochs=1, num_batch=4, batch_size=8)
log_artifact('./figures/results', artifact_path='figures')
log_model(gan, 'my_model', conda_env='./conda.yaml')

end_run()
