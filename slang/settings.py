"""To centralize settings for the Slang project."""

# TODO: Use config2py to make a config getter. This config getter should look at
#  environment variables, first, then .config/slang/configs folder,
#  the use the defaults within settings.py, then ask the user.
# TODO: Move all defaults of slang here

from config2py import get_app_data_folder

SLANG_DATA_DIR = get_app_data_folder('slang/data')
