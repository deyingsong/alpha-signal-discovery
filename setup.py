from setuptools import setup, find_packages

setup(
    name="quantitative-trading-strategy",
    version="1.0.0",
    description="A comprehensive quantitative trading strategy framework",
    author="Your Name",
    author_email="your.email@example.com",
    packages=find_packages(),
    install_requires=[
        "pandas>=1.5.0",
        "numpy>=1.23.0",
        "scikit-learn>=1.1.0",
        "xgboost>=1.6.0",
        "matplotlib>=3.5.0",
        "seaborn>=0.11.0",
        "plotly>=5.10.0",
        "openpyxl>=3.0.10",
        "pyyaml>=6.0",
        "scipy>=1.9.0",
        "statsmodels>=0.13.0",
        "jupyter>=1.0.0",
        "pytest>=7.0.0",
        "black>=22.0.0",
        "flake8>=5.0.0",
        "mypy>=0.971"
    ],
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Financial and Insurance Industry",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
)
