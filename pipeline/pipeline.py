import os.path, sys
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))

import kfp.dsl as dsl
from kfp.dsl import PipelineVolume

# To compile the pipeline:
#   dsl-compile --py pipeline.py --output pipeline.tar.gz
from src.constants import PROJECT_ROOT, CONDA_PYTHON_CMD


def git_clone_darkrai_op(repo_url: str):
    image = 'alpine/git:latest'

    commands = [
        f"git clone {repo_url} {PROJECT_ROOT}",
        f"cd {PROJECT_ROOT}"]

    volume_op = dsl.VolumeOp(
        name="create pipeline volume",
        resource_name="pipeline-pvc",
        modes=["ReadWriteOnce"],
        size="3Gi"
    )

    op = dsl.ContainerOp(
        name='git clone',
        image=image,
        command=['sh'],
        arguments=['-c', ' && '.join(commands)],
        container_kwargs={'image_pull_policy': 'IfNotPresent'},
        pvolumes={"/workspace": volume_op.volume}
    )

    return op


def preprocess_op(image: str, pvolume: PipelineVolume, data_dir: str):
    return dsl.ContainerOp(
        name='preprocessing',
        image=image,
        command=[CONDA_PYTHON_CMD, f"{PROJECT_ROOT}/src/preprocessing.py"],
        arguments=["--data_dir", data_dir],
        container_kwargs={'image_pull_policy': 'IfNotPresent'},
        pvolumes={"/workspace": pvolume}
    )


def train_and_eval_op(image: str, pvolume: PipelineVolume, data_dir: str, epochs: str):
    return dsl.ContainerOp(
        name='training and evaluation',
        image=image,
        command=[CONDA_PYTHON_CMD, f"{PROJECT_ROOT}/src/train.py"],
        arguments=["--data_dir", data_dir, "--epochs", epochs],
        file_outputs={
            'output': f'{PROJECT_ROOT}/output.txt',
            'cm':'/confusion_matrix.csv',
        },
        output_artifact_paths={
            'mlpipeline-metrics':'/mlpipeline-metrics.json',
            'mlpipeline-ui-metadata': '/mlpipeline-ui-metadata.json'
        },
        container_kwargs={'image_pull_policy': 'IfNotPresent'},
        pvolumes={"/workspace": pvolume}
    )


@dsl.pipeline(
    name='Fashion MNIST Training Pipeline',
    description='Fashion MNIST Training Pipeline to be executed on KubeFlow.'
)
def training_pipeline(image: str = 'shyhs/kubeflow-mnist',
                      repo_url: str = 'https://github.com/syhsu/kubeflow-mnist.git',
                      data_dir: str = '/workspace', epochs: str = '5'):
    git_clone = git_clone_darkrai_op(repo_url=repo_url)

    preprocess_data = preprocess_op(image=image,
                                    pvolume=git_clone.pvolume,
                                    data_dir=data_dir)

    _training_and_eval = train_and_eval_op(image=image,
                                           pvolume=preprocess_data.pvolume,
                                           data_dir=data_dir, epochs=epochs)


if __name__ == '__main__':
    import kfp.compiler as compiler
    version = '2.4'
    compiler.Compiler().compile(training_pipeline, 'Gitpipeline_v' + version + '.tar.gz')
