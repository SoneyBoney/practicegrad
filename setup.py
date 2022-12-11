import setuptools

setuptools.setup(
    name="practicegrad",
    version="0.0.0",
    author="Campinghedgehog",
    author_email="campinghedgehog@gmail.com",
    description="practice autograd engine",
    long_description_content_type="text/markdown",
    url="https://github.com/SoneyBoney/practicegrad",
    packages=setuptools.find_packages(),
    install_requires=["torch"],
    python_requires=">=3.8",
)
