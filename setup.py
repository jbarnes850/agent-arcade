from setuptools import setup, find_packages

setup(
    name="agent-arcade",
    version="0.1.0",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "ale-py>=0.10.2",
        "gymnasium[atari]>=0.29.1",
        "stable-baselines3[extra]>=2.5.0",
        "PyYAML>=6.0.1",
        "wandb>=0.15.0",
        "tensorboard>=2.15.1",
        "torch>=2.3.0",
        "numpy>=1.20.0",
        "loguru>=0.7.0",
        "pydantic>=2.0.0",
        "Click>=8.0",
        "pettingzoo[atari]>=1.24.1",
        "multi-agent-ale-py>=0.1.11"
    ],
    extras_require={
        "staking": [
            "py-near>=1.1.50",
            "base58>=2.1.1",
        ]
    },
    entry_points={
        "console_scripts": [
            "agent-arcade=cli.main:cli",
        ],
    },
    python_requires=">=3.9",
    author="Agent Arcade Team",
    description="Train and compete with AI agents using deep reinforcement learning",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/jbarnes850/agent-arcade",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
    ],
) 