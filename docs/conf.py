#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# xnns documentation build configuration file.
#
# The look and feel follows the MolSSI SEAMM documentation
# (https://molssi-seamm.github.io): pydata-sphinx-theme with the MolSSI
# palette in _static/css/custom.css and the MolSSI footer in
# _templates/molssi_footer.html.

import os
import sys

# Make the package importable without installation (docs/ -> repo root -> src/)
sys.path.insert(0, os.path.abspath(os.path.join("..", "src")))

# -- General configuration ---------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.githubpages",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx_design",
    "sphinx_copybutton",
    "sphinx_togglebutton",
    "myst_nb",
]

templates_path = ["_templates"]
source_suffix = {".rst": "restructuredtext"}
master_doc = "index"

project = "xnns"
copyright = "2026, The Molecular Sciences Software Institute"
author = "The Molecular Sciences Software Institute"

release = os.getenv("DOC_VERSION", "0.1.0")
version = release

exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

pygments_style = "default"

# -- Autodoc / autosummary ----------------------------------------------

autosummary_generate = True
autodoc_member_order = "bysource"
autodoc_typehints = "description"

napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_use_ivar = True
napoleon_use_param = True
napoleon_use_rtype = True

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "torch": ("https://pytorch.org/docs/stable/", None),
}

# -- Example notebooks (myst-nb) ----------------------------------------

# The example notebooks are committed fully executed, so the docs build only
# renders them -- it never runs them (no GPU, datasets, or extra venvs needed).
nb_execution_mode = "off"
# Training loops write progress across many separate stdout writes; without
# this each one becomes its own output block, breaking a single log into
# dozens of boxes.
nb_merge_streams = True
myst_enable_extensions = ["dollarmath", "amsmath", "html_image", "colon_fence"]

# Sphinx only reads sources inside docs/, while the notebooks live in
# examples/ at the repository root. Mirror them (plus the figures/ images some
# markdown cells reference) into docs/examples/nb/ when the build starts; the
# mirror is gitignored, and docs/examples/index.rst holds the gallery toctree.
def _mirror_example_notebooks():
    import shutil

    src = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "examples"))
    dst = os.path.join(os.path.dirname(__file__), "examples", "nb")

    def ignore(directory, names):
        drop = set()
        for name in names:
            path = os.path.join(directory, name)
            if os.path.isdir(path):
                if name in ("runs", "bamboo_upstream", ".ipynb_checkpoints"):
                    drop.add(name)
            elif not (name.endswith(".ipynb")
                      or os.path.basename(directory) == "figures"):
                drop.add(name)
        return drop

    shutil.copytree(src, dst, ignore=ignore, dirs_exist_ok=True)


_mirror_example_notebooks()

# -- Copy button --------------------------------------------------------

# Strip prompts when copying code cells
copybutton_prompt_text = r">>> |\.\.\. |\$ "
copybutton_prompt_is_regexp = True

# -- Options for HTML output -------------------------------------------

html_theme = "pydata_sphinx_theme"

html_theme_options = {
    # GitHub lives in icon_links below rather than in "github_url" -- setting
    # both renders the icon twice.
    "logo": {
        # Navbar brand: the MolSSI-AI mark, as in the e3nn course. The
        # molssi_* keys below are separate -- _templates/molssi_footer.html
        # reads those for the footer logo.
        "image_light": "molssi_ai_logo.png",
        "image_dark": "molssi_ai_logo.png",
        "molssi_light": "molssi_main_logo.png",
        "molssi_dark": "molssi_main_logo_inverted_white.png",
        "alt_text": "xnns - MolSSI-AI",
    },
    "announcement": (
        "xnns is under active development (pre-1.0): APIs may change between "
        "releases. Feedback and contributions are welcome."
    ),
    "show_toc_level": 2,
    # 0 makes the toctree captions themselves collapsible section headings in
    # the sidebar; at the default of 1 they are inert labels and every entry
    # under them is listed flat (87 of them under Examples).
    "show_nav_level": 0,
    "header_links_before_dropdown": 6,
    "external_links": [],
    # sidebar-secondary-collapse is ours (_templates/): pydata ships a collapse
    # button for the primary sidebar only. It goes first so it sits at the top
    # of the table of contents, mirroring the left-hand one.
    "secondary_sidebar_items": [
        "sidebar-secondary-collapse",
        "page-toc",
        "sourcelink",
    ],
    "footer_start": ["molssi_footer"],
    "footer_end": [],
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/molssi-ai/xnns",
            "icon": "fa-brands fa-github",
            "type": "fontawesome",
        },
        # A PyPI icon belongs here once the package is published; pypi.org
        # currently 404s for xnns, so linking it would be a dead icon.
        {
            "name": "MolSSI",
            "url": "https://molssi.org",
            "icon": "fa-solid fa-flask",
            "type": "fontawesome",
        },
    ],
}

html_static_path = ["_static"]
html_css_files = [
    "css/custom.css",
]

# Google Analytics 4. Deliberately not html_theme_options["analytics"]: that
# path hardcodes a consent default of analytics_storage='denied', which limits
# GA4 to cookieless pings and leaves the reports empty. See _static/gtag-init.js.
html_js_files = [
    ("https://www.googletagmanager.com/gtag/js?id=G-YK7FCTPX7M",
     {"async": "async"}),
    "gtag-init.js",
    "toc-collapse.js",
]

# The MolSSI-AI mark, matching the e3nn course.
html_favicon = "_static/molssi_ai_icon.png"

html_show_sphinx = False
html_show_copyright = False

htmlhelp_basename = "xnnsdoc"
