from setuptools import setup, find_packages

setup(
    name="text2sql-rag",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "transformers==4.31.0",
        "torch==2.0.1",
        "datasets==2.14.0",
        "sqlalchemy==2.0.19",
        "faiss-cpu==1.7.4",
        "python-dotenv==1.0.0",
        "fastapi==0.101.0",
        "uvicorn==0.23.2",
        "pydantic==2.1.1",
        "numpy==1.25.2",
        "pandas==2.0.3",
        "sentence-transformers==2.2.2",
        "pytest==7.4.0",
    ],
)
