import pathlib

def make_directory(path:pathlib.Path) -> None:
    """Makes the given directory if it does not exist.

    Args:
        path (pathlib.Path): the directory to create
    """
    path.mkdir(parents=True, exist_ok=True)

