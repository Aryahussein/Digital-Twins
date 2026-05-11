# docs/conf.py
import os
import sys

# Add the project root (or src folder) to sys.path so Sphinx can import your
# code if needed.
sys.path.insert(0, os.path.abspath('../../src'))

project = 'Digital Twins Course EDA'
author = 'UTD 7V88 Students'
release = '0.1'

extensions = [
    # You can add more extensions here if you need them
]

# General configuration
templates_path = ['_templates']
exclude_patterns = []

# HTML output options
html_theme = 'alabaster'          # or 'sphinx_rtd_theme', etc.
html_static_path = ['_static']
