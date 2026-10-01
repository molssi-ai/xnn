"""``xnn models``: list, inspect, download and package pre-trained models.

::

    xnn models list [--cached] [--format mace-torch] [--architecture mace]
    xnn models info mace-off23-small
    xnn models pull mace-mh-0 --head omat_pbe [--cache-dir /scratch/models]
    xnn models pack runs/exp/best.pt my-model/ --description "..." --archive
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path


def main(argv=None) -> None:
    """Entry point of ``xnn models``.

    Parameters
    ----------
    argv : list of str or None, optional
        Arguments after ``models``; ``None`` reads ``sys.argv[2:]``.
    """
    from ...benchmark.report import format_table
    from . import fetch_model, list_models, model_card, save_pretrained

    p = argparse.ArgumentParser(prog="xnn models",
                                description="Manage pre-trained models.")
    sub = p.add_subparsers(dest="action", required=True)

    ls = sub.add_parser("list", help="list registered and cached models")
    ls.add_argument("--cached", action="store_true",
                    help="only models that load without a download")
    ls.add_argument("--format", default=None, help="only this format (xnn, mace-torch)")
    ls.add_argument("--architecture", default=None, help="only this architecture")
    ls.add_argument("--tag", default=None, help="only models with this tag")
    ls.add_argument("--names", action="store_true", help="print names only")

    info = sub.add_parser("info", help="print a model's card")
    info.add_argument("name")

    pull = sub.add_parser("pull", help="download (and convert) a model into the cache")
    pull.add_argument("source", help="registered name, URL or Zenodo DOI")
    pull.add_argument("--head", default=None, help="head of a multi-head model")
    pull.add_argument("--filename", default=None, help="file of a multi-file Zenodo record")
    pull.add_argument("--format", default=None, help="format of the published file")
    pull.add_argument("--force", action="store_true", help="download again")

    pack = sub.add_parser("pack", help="write a checkpoint as a portable model directory")
    pack.add_argument("checkpoint", help="trainer checkpoint, model directory or foreign file")
    pack.add_argument("out", help="output directory")
    pack.add_argument("--name", default=None,
                      help="model name (default: the directory name); a registered "
                           "name fills in the card fields not given")
    pack.add_argument("--description", default="")
    pack.add_argument("--license", default=None)
    pack.add_argument("--citation", default=None)
    pack.add_argument("--tag", dest="tags", action="append", default=[])
    pack.add_argument("--archive", action="store_true",
                      help="also write <out>.zip, ready to upload")

    for sp in (ls, info, pull):
        sp.add_argument("--cache-dir", default=None,
                        help="model cache (default: $XNN_MODELS, else the "
                             "repository's models/ directory)")
    args = p.parse_args(sys.argv[2:] if argv is None else argv)

    if args.action == "list":
        rows = list_models(args.cache_dir, cached_only=args.cached, format=args.format,
                           architecture=args.architecture, tag=args.tag, details=True)
        if args.names:
            print("\n".join(r["name"] for r in rows))
        else:
            print(format_table(rows))
    elif args.action == "info":
        print(json.dumps(model_card(args.name, args.cache_dir).to_dict(), indent=2))
    elif args.action == "pull":
        print(fetch_model(args.source, cache_dir=args.cache_dir, head=args.head,
                          filename=args.filename, format=args.format,
                          force_download=args.force))
    elif args.action == "pack":
        fields = {k: v for k, v in dict(description=args.description, license=args.license,
                                         citation=args.citation, tags=args.tags).items() if v}
        if args.name:
            fields["name"] = args.name
            # a registered name lends its card to the fields not given here
            from .registry import get_card
            reg = get_card(args.name)
            if reg is not None:
                inherited = {k: v for k, v in reg.to_dict().items()
                             if k in ("description", "license", "citation", "tags", "units")}
                fields = {**inherited, **fields}
        out = save_pretrained(args.checkpoint, args.out, **fields)
        print("wrote", out)
        if args.archive:
            print("wrote", shutil.make_archive(str(out), "zip", root_dir=out.parent,
                                               base_dir=out.name))


if __name__ == "__main__":
    main()
