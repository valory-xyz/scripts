"""CLI tool that provides summative statistics for SpookySwap."""

from cli import create_parser


def start() -> None:
    """Runs the script."""
    args = create_parser().parse_args()
    # fetch(args)


if __name__ == '__main__':
    start()
