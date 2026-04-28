"""`chisel` CLI entrypoint.

Each subcommand is a thin wrapper around a tool or pipeline. As tools are
added, register them here so they're discoverable via `chisel --help`.
"""

import click

from protein_chisel import __version__


@click.group(help="protein_chisel — refine sequences for de novo enzyme designs.")
@click.version_option(version=__version__)
def main() -> None:
    pass


# ---- Tool subcommands (registered as they're implemented) -----------------
# Example skeleton; uncomment and flesh out in protein_chisel.tools.
#
# from protein_chisel.tools import classify_positions
# main.add_command(classify_positions.cli, name="classify-positions")


# ---- Pipeline subcommands -------------------------------------------------
# from protein_chisel.pipelines import enzyme_optimize_v1
# main.add_command(enzyme_optimize_v1.cli, name="enzyme-optimize-v1")


if __name__ == "__main__":
    main()
