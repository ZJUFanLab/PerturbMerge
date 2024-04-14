from setuptools import setup, find_packages

setup(
    name='perturbmerge',
    version='1.0.0',
    author='Junyun Cheng',
    author_email='ileslie@zju.edu.cn',
    description='PerturbMerge: Predicting the impact of gene perturbations (knockout or inhibition) on cell survival[Lol] and proliferation',
    long_description='PerturbMerge is a Python package that predicts the impact of 485 specific gene knockouts on cell viability by analyzing single-cell transcriptomic data. It integrates single-cell transcriptomic sequencing data from 164 undisturbed cancer cell lines, along with corresponding essential gene screening information, and is trained using three ensemble learning algorithms: Random Forest (RF), Gradient Boosting Machine (GBM), and XGBoost. This results in a comprehensive classifier capable of predicting the effects of specific gene knockouts on the survival and proliferation of single cells based on the input data.',
    long_description_content_type='text/markdown',
    url='https://github.com/ZJUFanLab/PerturbMerge',
    packages=find_packages(),
    classifiers=[
        'Development Status :: 3 - Alpha',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.9',
    ],
    python_requires='>=3.9',
    install_requires=[
        'joblib==1.3.2',
        'numpy==1.26.4',
        'pandas==2.2.2',
        'scikit-learn==1.2.0',
        'xgboost==2.0.3',
        'tqdm==4.66.2'
    ],
)