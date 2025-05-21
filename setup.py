#!/usr/bin/env python
"""
Setup script for BTC-AI package
"""

from setuptools import setup, find_packages
import os
import re

# Read version from __init__.py
with open(os.path.join('src', '__init__.py'), 'r') as f:
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", f.read(), re.M)
    version = version_match.group(1) if version_match else '0.1.0'

# Read long description from README
with open('README.md', 'r', encoding='utf-8') as f:
    long_description = f.read()

# Read requirements from requirements.txt
with open('requirements.txt', 'r') as f:
    requirements = f.read().splitlines()

setup(
    name="btc-ai",
    version=version,
    author="Your Name",
    author_email="your.email@example.com",
    description="Reinforcement learning system for Bitcoin trading",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/BTC-AI",
    package_dir={"": "src"},
    packages=find_packages("src"),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Financial and Insurance Industry",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Office/Business :: Financial :: Investment",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.7",
    install_requires=requirements,
    entry_points={
        'console_scripts': [
            'btc-ai=src.ui.main:main',
            'btc-ai-train=src.training.training:main',
            'btc-ai-backtest=src.training.backtesting:main',
        ],
    },
    include_package_data=True,
    package_data={
        "btc-ai": ["configs/*.json", "docs/*.md"],
    },
) 