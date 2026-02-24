# setup.py
from setuptools import setup, find_packages
import pathlib

HERE = pathlib.Path(__file__).parent
LONG_DESC = (HERE / "README.md").read_text(encoding="utf-8")

setup(
    name="deepseek-replica",
    version="0.1.0",
    description="From-scratch PyTorch replica of the DeepSeek LLM architecture",
    long_description=LONG_DESC,
    long_description_content_type="text/markdown",
    author="mayankbot01",
    url="https://github.com/mayankbot01/deepseek-replica",
    license="MIT",
    python_requires=">=3.10",
    packages=find_packages(exclude=["tests", "tests.*", "scripts", "configs"]),
    install_requires=[
        "torch>=2.1.0",
        "numpy>=1.24",
        "tqdm>=4.65",
        "pyyaml>=6.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0",
            "black",
            "isort",
            "flake8",
        ],
        "data": [
            "datasets>=2.14",
            "transformers>=4.36",
            "sentencepiece>=0.1.99",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    entry_points={
        "console_scripts": [
            "deepseek-train=train:main",
            "deepseek-eval=evaluate:main",
            "deepseek-infer=inference:main",
        ]
    },
)
