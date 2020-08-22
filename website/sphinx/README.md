# sphinx API reference

This file describes the sphinx setup for auto-generating the API reference.


## Installation

**Requirements**:
- sphinx >= 2.0
- sphinx_autodoc_typehints

You can install these via `pip install sphinx sphinx_autodoc_typehints`.


## Building

From the `website/sphinx` directory, run `make html`.

Generated HTML output can be found in the `website/sphinx/build` directory. The main index page is: `website/sphinx/build/html/index.html`


## Structure

`source/index.rst` contains the main index. The API reference for each module lives in its own file.
