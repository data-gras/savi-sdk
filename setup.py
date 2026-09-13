from pathlib import Path
from setuptools import setup, find_packages

long_description = (Path(__file__).parent / "README.md").read_text(encoding="utf-8")

setup(
    name="savi-sdk",
    version="0.15.1",
    description="Lightweight observability SDK for LLM costs, carbon, and compliance",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="SAVI",
    author_email="contact@datagras.com",
    url="https://github.com/data-gras/savi-sdk",
    project_urls={
        "Documentation": "https://github.com/data-gras/savi-sdk#readme",
        "Bug Tracker":   "https://github.com/data-gras/savi-sdk/issues",
        "Changelog":     "https://github.com/data-gras/savi-sdk/releases",
    },
    license="MIT",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: System :: Monitoring",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    keywords=[
        "llm", "ai", "finops", "observability", "cost", "openai", "anthropic",
        "bedrock", "cohere", "mistral", "vertex", "azure", "carbon", "compliance",
        "pii", "aud", "savi",
    ],
    packages=find_packages(exclude=["tests*"]),
    python_requires=">=3.11",
    install_requires=["httpx>=0.27.0"],
    extras_require={
        "dev":     ["pytest>=8.2.0", "pytest-asyncio>=0.23.7", "openai>=1.0.0", "anthropic>=0.25.0", "mcp>=1.0.0"],
        "openai":  ["openai>=1.0.0"],
        "anthropic": ["anthropic>=0.25.0"],
        "google":  ["google-cloud-aiplatform>=1.55.0", "vertexai>=1.55.0"],
        "azure":   ["openai>=1.0.0"],
        "bedrock": ["boto3>=1.34.0"],
        "cohere":  ["cohere>=5.3.0"],
        "mistral": ["mistralai>=1.0.0"],
        "mcp":     ["mcp>=1.0.0"],
        "pii":     ["presidio-analyzer>=2.2.0", "presidio-anonymizer>=2.2.0", "spacy>=3.0.0"],
        "all":     [
            "openai>=1.0.0", "anthropic>=0.25.0",
            "google-cloud-aiplatform>=1.55.0", "vertexai>=1.55.0",
            "boto3>=1.34.0",
            "cohere>=5.3.0",
            "mistralai>=1.0.0",
            "mcp>=1.0.0",
            "presidio-analyzer>=2.2.0", "presidio-anonymizer>=2.2.0", "spacy>=3.0.0",
        ],
    },
)
