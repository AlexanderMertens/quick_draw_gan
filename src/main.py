import tensorflow as tf

from mlflow.tracking.fluent import end_run, log_artifact, start_run

from models.train_model import train_model

run_name = 'WGAN-v1-filtered-envelope'
start_run(run_name=run_name)

gan = train_model(num_epochs=10, n_disc=5, num_batch=1,
                  batch_size=64, run_name=run_name)
log_artifact('./figures/results', artifact_path='figures')

end_run()
