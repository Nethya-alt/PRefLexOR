from setuptools import setup, find_packages

setup(
    name='PRefLexOR',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'torch',
        'transformers',
        'matplotlib',
        'tqdm',
    ],
    entry_points={
        'console_scripts': [
            # Add any console scripts here, if applicable
        ],
    },
    description='PRefLexOR: Recursive Language Modeling for Exploratory Reasoning Optimization',
    author='Your Name',
    url='https://github.com/lamm-mit/PRefLexOR',
)
