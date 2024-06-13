import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="lautcomponents",
    version="1.1.1",
    author="lautmalers",
    author_email="info@die-lautmaler.com",
    description="lautmaler in house rasa components",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/die-lautmaler/rasa_components",
    project_urls={
        "Bug Tracker": "https://github.com/die-lautmaler/rasa_components/issues"
    },
    # license='MIT',
    packages=["lautcomponents"],
    install_requires=["rasa>=3.0.0", "black>=22.0.0", "tokenizers", "websockets==10.0"],
)
