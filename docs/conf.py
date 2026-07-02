# Configuration file for the Sphinx documentation builder.
# -- Project information -----------------------------------------------------

project = "LLM-as-a-Verifier"
copyright = "2026, LLM-as-a-Verifier Team"
author = "LLM-as-a-Verifier Team"

version = "0.1.0"
release = version

# -- General configuration ---------------------------------------------------

extensions = [
    "myst_parser",
    "sphinx.ext.mathjax",
    "sphinx.ext.intersphinx",
    "sphinx_copybutton",
]

myst_enable_extensions = [
    "dollarmath",
    "amsmath",
    "colon_fence",
    "deflist",
    "html_image",
]
myst_heading_anchors = 3

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

# -- Options for HTML output -------------------------------------------------

html_title = "LLM-as-a-Verifier"
html_theme = "sphinx_book_theme"
html_logo = "_static/image/logo.png"
html_favicon = "_static/image/logo.png"
html_static_path = ["_static"]

html_theme_options = {
    "repository_url": "https://github.com/llm-as-a-verifier/llm-as-a-verifier",
    "use_repository_button": True,
    "use_issues_button": True,
    "use_edit_page_button": True,
    "path_to_docs": "docs",
    "show_navbar_depth": 1,
    "show_toc_level": 2,
    "home_page_in_toc": True,
}

html_context = {
    "display_github": True,
    "github_user": "llm-as-a-verifier",
    "github_repo": "llm-as-a-verifier",
    "github_version": "main",
    "conf_py_path": "/docs/",
}

# -- Options for sphinx-copybutton -------------------------------------------

copybutton_prompt_text = r">>> |\.\.\. |\$ "
copybutton_prompt_is_regexp = True
