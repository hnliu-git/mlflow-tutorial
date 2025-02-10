
import mlflow
from mlflow.utils.git_utils import get_git_commit, get_git_branch, get_git_repo_url

def setup_git_info():

    mlflow.set_tag("mlflow.source.name", get_git_repo_url('.'))
    mlflow.set_tag("mlflow.source.git.commit", get_git_commit('.'))
    mlflow.set_tag("mlflow.source.git.repoURL", get_git_repo_url('.'))
    mlflow.set_tag("mlflow.source.git.branch", get_git_branch('.'))


if __name__ == '__main__':
    with mlflow.start_run() as run:
        setup_git_info()