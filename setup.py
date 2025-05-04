from setuptools import setup, find_packages

setup(
    name="agent-team",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "jinja2>=3.0.0",
        "requests>=2.25.1",
        "python-dotenv>=0.19.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-mock>=3.10.0",
            "pytest-cov>=4.0.0",
        ]
    },
    author="DeeNihl",
    author_email="deenihl@github.com",
    description="A bare bones Python autonomous agent project",
    keywords="ai, agents, autonomous, team",
    python_requires=">=3.7",
)