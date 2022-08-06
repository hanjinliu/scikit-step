from setuptools import setup, find_packages

with open("skstep/__init__.py", encoding="utf-8") as f:
    line = next(iter(f))
    VERSION = line.strip().split()[-1][1:-1]

with open("README.md") as f:
    readme = f.read()

setup(
    name="scikit-step",
    version=VERSION,
    description="1-D step detection algorithms",
    long_description=readme,
    long_description_content_type="text/markdown",
    author="Hanjin Liu",
    author_email="liuhanjin-sc@g.ecc.u-tokyo.ac.jp",
    license="BSD 3-Clause",
    download_url="https://github.com/hanjinliu/scikit-step",
    packages=find_packages(exclude=["docs", "examples", "rst", "tests", "tests.*"]),
    install_requires=[
        "numpy>=1.21",
        "scipy>=1.7.3",
    ],
    extras_require={"all": ["dask>=2021.11.1", "matplotlib>=3.1"]},
    python_requires=">=3.8",
)
