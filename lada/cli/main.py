# SPDX-FileCopyrightText: Lada Authors
# SPDX-License-Identifier: AGPL-3.0

import argparse
from gettext import gettext as _
import sys

from lada import VERSION
from lada.cli import utils
from lada.cli.compile_command import add_compile_subparser
from lada.cli.restore_command import add_restore_subparser


def setup_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        usage=_('%(prog)s <command> [options]'),
        description=_("Restore pixelated adult videos (JAV)"),
        formatter_class=utils.TranslatableHelpFormatter,
        add_help=False,
    )

    group_general = parser.add_argument_group(_("General"))
    group_general.add_argument("--version", action="store_true", help=_("Display version and exit"))
    group_general.add_argument("--help", action="store_true", help=_("Show this help message and exit"))

    subparsers = parser.add_subparsers(dest="command")
    add_restore_subparser(subparsers)
    add_compile_subparser(subparsers)

    return parser


def main():
    argparser = setup_argparser()
    argv = sys.argv[1:]
    if not argv:
        argparser.print_help()
        return
    if argv[0] == "--help":
        argparser.print_help()
        return
    if argv[0] == "--version":
        print("Lada: ", VERSION)
        return
    args = argparser.parse_args(argv)
    args.func(args)


if __name__ == '__main__':
    main()
