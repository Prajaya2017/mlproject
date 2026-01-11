from setuptools import find_packages,setup
from typing import List
import os

HYPEN_E_DOT = '-e .'

def get_requirements(file_path: str) -> List[str]:
    requirements = []
    
    if not os.path.exists(file_path):
        return requirements
    
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.replace("\n","") for req in requirements]
        
        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)
    
    return requirements

setup(
    name='mlproject',
    version='0.0.1',
    author='Prajaya',
    author_email='prajaya2009@gmail.com',
    packages=find_packages(),  # Use 'praj' as package name
    install_requires=get_requirements('requirements.txt')
)
