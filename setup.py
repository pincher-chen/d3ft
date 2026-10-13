from setuptools import setup,find_packages
from pathlib import Path
setup(name='d3ft',version='0.1.0',description='Direct free-energy minimization in plane-wave DFT',license='Apache-2.0',packages=find_packages(include=['d3ft','d3ft.*']),python_requires='>=3.10',install_requires=Path('requirements.txt').read_text().splitlines(),entry_points={'console_scripts':['d3ft=d3ft.cli:main']})
