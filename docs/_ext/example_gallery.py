"""Gate the example gallery: published pages link, the rest read "coming soon".

The ``example-toctree`` directive lists notebook pages like ``toctree`` does,
but only the published ones become links (and enter the navigation through
a hidden toctree). Every other entry is rendered as muted text with a
"coming soon" marker, and its notebook, if present, is left out of the build,
so an unreviewed page is neither linked nor reachable.

A page is published when its notebook is part of the checkout: the
notebooks under ``examples/`` are kept out of the repository until they are
reviewed (see ``.gitignore``), and committing one publishes it. The
``examples_ready`` configuration value overrides that rule with an explicit
list of pages (paths relative to ``docs/examples/`` without the ``.ipynb``
suffix), which is how a local build with every notebook on disk can preview
the gallery as the published site shows it.

Usage in ``docs/examples/index.rst``::

    .. example-toctree::
       :caption: ANI (dnn)

       Training ANI from scratch on rMD17 <nb/dnn/ani/ani_rmd17_train>
       nb/dnn/ani/ani1_dataset

The title before ``<...>`` is what a pending entry shows (inline code spans
and ``$math$`` are rendered); it is required when the notebook is not in the
checkout and ignored once the page is published, which then carries the
notebook's own heading. An entry without a title takes the heading of the
notebook on disk.
"""
from __future__ import annotations

import json
import os
import posixpath
import re

from docutils import nodes
from docutils.parsers.rst import directives
from docutils.statemachine import StringList
from sphinx.application import Sphinx
from sphinx.directives.other import TocTree
from sphinx.util import logging
from sphinx.util.docutils import SphinxDirective

logger = logging.getLogger(__name__)

EXAMPLES_DIR = "examples"
PENDING_LABEL = "coming soon"
_TITLED_ENTRY = re.compile(r"^(.+?)\s*<(.+?)>$")


class example_link(nodes.General, nodes.Element):
    """Placeholder for the link to a published page, resolved once titles are known."""


def _join(base_docname: str, entry: str) -> str:
    if entry.startswith("/"):
        return posixpath.normpath(entry[1:])
    return posixpath.normpath(posixpath.join(posixpath.dirname(base_docname), entry))


def _split_entry(entry: str) -> tuple[str | None, str]:
    match = _TITLED_ENTRY.match(entry)
    if match:
        return match.group(1).strip(), match.group(2).strip()
    return None, entry


def _ready_list(config) -> list[str] | None:
    ready = config.examples_ready
    if ready is None:
        return None
    if isinstance(ready, str):
        ready = [r for r in ready.split(",") if r.strip()]
    return [r.strip() for r in ready]


def _ready_docnames(config) -> set[str] | None:
    ready = _ready_list(config)
    if ready is None:
        return None
    return {posixpath.normpath(posixpath.join(EXAMPLES_DIR, r)) for r in ready}


def _notebook_title(path: str) -> str | None:
    with open(path, encoding="utf-8") as fh:
        nb = json.load(fh)
    for cell in nb.get("cells", []):
        if cell.get("cell_type") != "markdown":
            continue
        source = cell.get("source", "")
        lines = source if isinstance(source, list) else source.splitlines(True)
        for line in lines:
            if line.startswith("# "):
                return line[2:].strip()
    return None


def _markdown_inline_to_rst(text: str) -> str:
    """Convert the inline markup a notebook heading uses (code spans, math) to RST."""
    text = re.sub(r"`([^`]+)`", r"``\1``", text)
    text = re.sub(r"\$([^$]+)\$", r":math:`\1`", text)
    return text


class ExampleToctree(SphinxDirective):
    has_content = True
    option_spec = {
        "caption": directives.unchanged_required,
    }

    def run(self):
        ready = _ready_docnames(self.config)
        srcdir = str(self.env.srcdir)
        entries = [line.strip() for line in self.content if line.strip()]

        listing = nodes.bullet_list(classes=["example-toctree"])
        published: list[str] = []
        for entry in entries:
            title, path = _split_entry(entry)
            docname = _join(self.env.docname, path)
            notebook = os.path.join(srcdir, docname + ".ipynb")
            present = os.path.exists(notebook)
            is_ready = present if ready is None else docname in ready

            item = nodes.list_item(classes=["toctree-l1"])
            para = nodes.paragraph()
            if is_ready:
                if not present:
                    logger.warning("example-toctree: %r is published but %s does not exist",
                                   path, notebook, location=(self.env.docname, self.lineno))
                    continue
                para += example_link(docname=docname)
                published.append(path)
            else:
                if title is None and present:
                    title = _notebook_title(notebook)
                if title is None:
                    logger.warning("example-toctree: %r has no title and no notebook to read "
                                   "one from; write the entry as 'Title <%s>'", path, path,
                                   location=(self.env.docname, self.lineno))
                    continue
                text_nodes, messages = self.state.inline_text(
                    _markdown_inline_to_rst(title), self.lineno)
                pending = nodes.inline(classes=["example-pending"])
                pending += text_nodes
                pending += nodes.inline(PENDING_LABEL, PENDING_LABEL,
                                        classes=["example-pending-marker"])
                para += pending
                para += messages
            item += para
            listing += item

        wrapper = nodes.compound(classes=["toctree-wrapper", "example-gallery"])
        wrapper += listing
        result: list[nodes.Node] = [wrapper]

        if published:
            options = {"hidden": None, "maxdepth": 1}
            if "caption" in self.options:
                options["caption"] = self.options["caption"]
            toctree = TocTree("toctree", [], options,
                              StringList(published, source=self.content.source(0)),
                              self.lineno, self.content_offset, self.block_text,
                              self.state, self.state_machine)
            result.extend(toctree.run())
        return result


def _exclude_pending_notebooks(app: Sphinx, config) -> None:
    """With an explicit ``examples_ready`` list, keep the other notebooks out of the build."""
    listed = _ready_list(config)
    if listed is None:
        return
    root = os.path.join(str(app.srcdir), EXAMPLES_DIR)
    ready = {posixpath.normpath(r) for r in listed}
    found: set[str] = set()
    for dirpath, _dirs, files in os.walk(root):
        for name in files:
            if not name.endswith(".ipynb"):
                continue
            rel = os.path.relpath(os.path.join(dirpath, name), root)
            rel = rel.replace(os.sep, "/")[: -len(".ipynb")]
            found.add(rel)
            if rel not in ready:
                config.exclude_patterns.append(posixpath.join(EXAMPLES_DIR, rel + ".ipynb"))
    for missing in sorted(ready - found):
        logger.warning("examples_ready names %r but %s/%s.ipynb does not exist",
                       missing, EXAMPLES_DIR, missing)


def _resolve_links(app: Sphinx, doctree: nodes.document, fromdocname: str) -> None:
    for node in list(doctree.findall(example_link)):
        docname = node["docname"]
        title = app.env.titles.get(docname)
        if title is None:
            logger.warning("example-toctree: %r has no title (not built?)", docname,
                           location=node)
            node.replace_self(nodes.Text(docname))
            continue
        ref = nodes.reference("", "", internal=True,
                              refuri=app.builder.get_relative_uri(fromdocname, docname),
                              classes=["reference", "internal"])
        ref += [child.deepcopy() for child in title.children]
        node.replace_self(ref)


def setup(app: Sphinx):
    app.add_config_value("examples_ready", None, "env", [list, str, type(None)])
    app.add_node(example_link)
    app.add_directive("example-toctree", ExampleToctree)
    app.connect("config-inited", _exclude_pending_notebooks)
    app.connect("doctree-resolved", _resolve_links)
    return {"version": "2", "parallel_read_safe": True, "parallel_write_safe": True}
