from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="ecg-arrhythmia-pipeline",
    version="1.0.0",
    author="Eira (Iqra Nizam)",
    author_email="iqra.nizam@example.com",
    description="Production-grade 12-lead ECG classification pipeline (Zheng et al. 2020)",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/iqranizam/ecg-arrhythmia-pipeline",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Healthcare Industry",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
    ],
    python_requires=">=3.7",
    install_requires=[
        "numpy>=1.21.0",
        "scipy>=1.7.0",
        "pandas>=1.3.0",
        "scikit-learn>=1.0.0",
        "xgboost>=1.5.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.2.0",
            "pytest-cov>=2.12.0",
            "black>=21.0",
            "flake8>=3.9.0",
        ],
        "viz": [
            "matplotlib>=3.4.0",
            "seaborn>=0.11.0",
            "jupyter>=1.0.0",
        ],
    },
    keywords="ECG arrhythmia classification signal-processing machine-learning xgboost",
    project_urls={
        "Bug Reports": "https://github.com/iqranizam/ecg-arrhythmia-pipeline/issues",
        "Source": "https://github.com/iqranizam/ecg-arrhythmia-pipeline",
        "Paper": "https://doi.org/10.1038/s41598-020-59821-7",
    },
)
