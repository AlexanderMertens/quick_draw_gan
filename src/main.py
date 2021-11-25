import tensorflow as tf

from mlflow.tracking.fluent import end_run, log_artifact, start_run

from models.train_model import train_model

tf.config.threading.set_intra_op_parallelism_threads(2)
tf.config.threading.set_inter_op_parallelism_threads(2)
run_name = 'GAN-v1-envelope'
start_run(run_name=run_name)

gan = train_model(num_epochs=100, num_batch=64,
                  batch_size=128, run_name=run_name)
log_artifact('./figures/results', artifact_path='figures')

end_run()
