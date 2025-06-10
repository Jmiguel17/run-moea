from setuptools import setup, find_packages

setup(
    name='run-moea',
    version='0.1',
    description='Run Borg and Platypus',
    url='',
    author='Jose M. Gonzalez',
    author_email='jose.gonzalezcabrera@manchester.ac.uk',
    packages=find_packages(),
    package_data={
        'run-moea': ['json/*.json'],
    },
    entry_points={
        'console_scripts': ['run-moea=run_moea.cli:start_cli'],
    }
)