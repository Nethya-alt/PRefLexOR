from setuptools import setup, find_packages

# Function to read requirements.txt
def parse_requirements(filename):
    with open(filename, 'r') as req_file:
        return req_file.read().splitlines()

setup(
    name='PRefLexOR',
    version='0.1.0',
    packages=find_packages(),
    install_requires=parse_requirements('requirements.txt'),
    description='PRefLexOR: Recursive Language Modeling for Exploratory Reasoning Optimization',
    author='Markus J. Buehler',
    url='https://github.com/lamm-mit/PRefLexOR',
)
