from __future__ import absolute_import, division, print_function

import os
import subprocess

from setuptools import find_packages, setup


# Function to update submodules
def update_submodules():
    if os.path.isdir(".git"):
        try:
            subprocess.check_call(["git", "submodule", "update", "--init", "--recursive"])
        except subprocess.CalledProcessError as e:
            print(f"Failed to update submodules: {e}")
        else:
            print("Submodules updated successfully.")
    else:
        print("Not a git repository.")


# Call the function to update submodules
update_submodules()

description = """SMAC-PettingZoo - Latest StarCraft Multi-Agent Challenge. The origin environment can be found at https://github.com/oxwhirl/smac.git and https://github.com/oxwhirl/smacv2.git."""

setup(
    name="smac_pettingzoo",
    version="1.0.0",
    description="SMAC-PettingZoo - Latest StarCraft Multi-Agent Challenge",
    long_description=description,
    license="MIT License",
    keywords="StarCraft, PettingZoo, Multi-Agent Reinforcement Learning",
    packages=find_packages(),
    install_requires=[
        "pysc2>=3.0.0",
        "protobuf<3.21",
        "s2clientprotocol>=4.10.1.75800.0",
        "absl-py>=0.1.0",
        "numpy>=1.10",
        "pygame>=2.0.0",
        "pettingzoo",
        "gymnasium",
    ],
)
