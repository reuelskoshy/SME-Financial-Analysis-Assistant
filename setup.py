from setuptools import setup, find_packages

setup(
    name="sme-analytics",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "langchain-ollama>=1.0.0",
        "langchain-core>=1.0.0",
        "faiss-cpu",
        "numpy",
        "pandas",
        "torch",
        "streamlit",
    ],
)