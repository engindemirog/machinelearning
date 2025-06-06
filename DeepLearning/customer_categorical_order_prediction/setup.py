from setuptools import setup, find_packages

setup(
    name="customer_category_prediction",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.26.0",
        "pandas>=2.1.0",
        "scikit-learn>=1.3.2",
        "tensorflow>=2.15.0",
        "psycopg2-binary>=2.9.9",
        "python-dotenv>=1.0.0",
        "pytest>=7.4.3",
        "black>=23.11.0",
        "flake8>=6.1.0",
        "mypy>=1.7.0",
        "matplotlib>=3.8.0",
        "seaborn>=0.13.0",
        "sqlalchemy>=2.0.0"
    ],
    python_requires=">=3.12.0",
) 