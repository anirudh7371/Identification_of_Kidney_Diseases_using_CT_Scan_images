import setuptools

with open("README.md" , "r", encoding="utf-8") as f:
    long_description = f.read()

__version__ = "1.0.0"

REPO_NAME = "Identification_of_kidney_stones_empowered_with_XAI"
AUTHOR_USER_NAME = "Anirudh"
SRC_REPO = "KidneyStoneClassification"
AUTHOR_EMAIL = "anirudh7371@gmail.com"

setuptools.setup(
    name = SRC_REPO,
    version = __version__,
    description= "Identification of Kidney Stones using X-Ray Images and empowering it with a layer of Explainable Artificial Intelligence",
    author=AUTHOR_USER_NAME,
    author_email=AUTHOR_EMAIL,
    url=f"https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}",
    package_dir={"":"src"},
    packages=setuptools.find_packages(where="src")
)
