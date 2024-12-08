from setuptools import setup, find_namespace_packages

setup(
    name="nih_cxr_ai",
    packages=find_namespace_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.8",
    install_requires=[
        "numpy",
        "pandas",
        "matplotlib",
        "seaborn",
        "scikit-learn",
        "torch>=2.0.0",
        "torchvision",
        "pytorch-lightning",
        "wandb",
        "torchmetrics",
        "datasets",
        "tqdm",
        "PyYAML"
    ],
    extras_require={
        "dev": [
            "pytest",
            "black",
            "flake8",
            "jupyter",
        ],
    },
    description="Deep learning for chest X-ray classification",
    author="Kulsoom Abdullah",
)



# from setuptools import find_packages, setup
# from setuptools import setup, find_namespace_packages



# setup(
#     name="nih_cxr_ai",
#     packages=find_namespace_packages(include=["src.*"]),
#     package_dir={"": "src"},
#     python_requires=">=3.8",
#     install_requires=[
#         "numpy",
#         "pandas",
#         "matplotlib",
#         "seaborn",
#         "scikit-learn",
#         "torch>=2.0.0",
#         "torchvision",
#         "pytorch-lightning",
#         "wandb",
#         "torchmetrics",
#     ],
#     extras_require={
#         "dev": [
#             "pytest",
#             "black",
#             "flake8",
#             "jupyter",
#         ],
#     },
#     description="Deep learning for chest X-ray classification",
#     author="Kulsoom Abdullah",
# )



