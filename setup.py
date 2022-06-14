import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name='laut_components',
    version='0.5.0',
    author='lautmalers',
    author_email='info@die-lautmaler.com',
    description='lautmaler in house rasa components',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/die-lautmaler/rasa_components',
    project_urls = {
        "Bug Tracker": "https://github.com/die-lautmaler/rasa_components/issues"
    },
    # license='MIT',
    packages=['laut_components'],
    install_requires=['rasa>=2.8.0,<3.0.0'],
)