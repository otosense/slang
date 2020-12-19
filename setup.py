from setuptools import setup, find_packages


def text_of_readme_md_file():
    try:
        with open('README.md') as f:
            return f.read()
    except:
        return ""


setup(
    packages=find_packages(),
    long_description=text_of_readme_md_file(),
    long_description_content_type="text/markdown"
)  # Note: Everything should be in the local setup.cfg
