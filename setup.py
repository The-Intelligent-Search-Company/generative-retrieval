#!/usr/bin/env python3
"""Setup script for Generative Retrieval Library."""

from setuptools import setup, find_packages

# Read README for long description
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read requirements
with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="generative-retrieval",
    version="0.1.0",
    author="Asad Khan",
    author_email="asad@theintelligentsearch.com",
    description="A production-ready library for generative retrieval systems and neural information retrieval",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/The-Intelligent-Search-Company/generative-retrieval",
    project_urls={
        "Bug Reports": "https://github.com/The-Intelligent-Search-Company/generative-retrieval/issues",
        "Source": "https://github.com/The-Intelligent-Search-Company/generative-retrieval",
        "Documentation": "https://github.com/The-Intelligent-Search-Company/generative-retrieval/blob/main/README.md",
    },
    packages=find_packages(exclude=["tests*", "examples*", "docs*"]),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Text Processing :: Indexing",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=22.0",
            "flake8>=4.0",
            "isort>=5.0",
            "mypy>=0.900",
            "jupyter>=1.0",
            "matplotlib>=3.0",
            "seaborn>=0.11",
            "ipywidgets>=7.0",
        ],
        "notebooks": [
            "jupyter>=1.0",
            "matplotlib>=3.0",
            "seaborn>=0.11",
            "ipywidgets>=7.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "gr-train=generative_retrieval.cli:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
    keywords="generative retrieval, neural information retrieval, DSI, differentiable search index, transformers, T5",
)