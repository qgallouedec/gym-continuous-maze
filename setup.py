from setuptools import find_packages, setup

setup(
    name="gym_continuous_maze",
    packages=find_packages(),
    install_requires=["gym", "pygame"],
    description="Continuous maze environment integrated with OpenAI/Gym",
    author="Quentin Gallouédec",
    url="https://github.com/qgallouedec/gym-continuous-maze",
    author_email="gallouedec.quentin@gmail.com",
    license="MIT",
    version="0.0.0",
)
