from setuptools import setup, find_packages


def readme():
    with open('README.md') as f:
        return f.read()


setup(
    name='slang',
    version='0.0.1',
    description='Tools to endow signals with language-like structure.',
    long_description=readme(),
    long_description_content_type="text/markdown",
    url='https://github.com/thorwhalen/slang',
    author='Thor Whalen',
    license='Apache',
    packages=find_packages(),
    install_requires=['numpy', 'pandas'],
    include_package_data=True,
    zip_safe=False,
    # download_url='https://github.com/i2mint/py2store/archive/v0.0.5.zip',
    keywords=['sound recognition', 'machine learning', 'language', 'audio', 'signal processing',
              'natural language processing', 'NLP', 'text mining'],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Topic :: Software Development',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python :: 3.7',
    ],
)
