from setuptools import setup, find_packages
setup(
    name='pycaML',
    version='0.3.1',
    author='Donato Riccio',
    description='Python Comparative Analysis for Machine Learning',
    long_description='pycaML is an easy machine learning model comparison tool with optimization. It allows to generate a table comparing multiple machine learning models, to see which one is best for your data. The unique feature of pycaML is built-in hyperparameters tuning using Bayesian Optimization. It also supports meta-models like Stacking and Voting ensembles. You can setup and optimize 25 models with one line of code.',
    url='https://github.com/reese3222/pycaML',
    keywords='machine learning, optimization, stacking',
    python_requires='>=3.7, <4',
    install_requires=['numpy', 'pandas', 'scikit-learn', 'scipy', 'xgboost', 'lightgbm', 'catboost', 'hyperopt', 'tqdm', 'interpret','hyperopt'],
   # package_dir={'pycaML': 'src'},
    packages=find_packages()
    )