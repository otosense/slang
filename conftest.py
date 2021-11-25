import os
import pathlib


def pytest_ignore_collect(path, ignore=('scrap', 'examples')):
    this_dir = pathlib.Path(__file__)
    root_dir = this_dir.parent.resolve()
    project_name = root_dir.name
    ignore_dirs = [
        pathlib.PurePath(root_dir, project_name, ignore_dir) for ignore_dir in ignore
    ]
    return any(str(ignore_dir) in str(path) for ignore_dir in ignore_dirs)
