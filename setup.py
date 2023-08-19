from setuptools import setup, find_packages
from typing import List


PROJECT_NAME="Machine Learning Project"
VERSION='0.0.1'
DESCRIPTION='This is my modular coding project'
AUTHOR="ALOK DWIVEDI"
AUTHOR_MAIL='Alokdwivedifgiet@gmail.com'
REQUIREMENTS='requirements.txt'
HYPHEN_E_DOT="-e ."

def Get_requirement_list()->List[str]:
    with open(REQUIREMENTS) as requirement_file:
        requirement_list=requirement_file.readlines()
        requirement_list=[requirement_name.replace("\n","") for requirement_name in requirement_list]
        if HYPHEN_E_DOT in requirement_list:
            requirement_list.remove(HYPHEN_E_DOT)

        return requirement_list

    


setup(name=PROJECT_NAME,
      version=VERSION,
      description=DESCRIPTION,
      author=AUTHOR,
      author_email=AUTHOR_MAIL,
      packages=find_packages(),
      install_requirement=Get_requirement_list()
     )