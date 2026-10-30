"""Start the official StarVLA policy server with full-checkpoint loading."""

from __future__ import annotations

import logging

from benchmarks.libero_official.starvla_runtime.starvla_full_checkpoint import (
    install_full_checkpoint_loader,
)


def main() -> None:
    install_full_checkpoint_loader()

    from deployment.model_server.server_policy import build_argparser
    from deployment.model_server.server_policy import main as serve

    logging.basicConfig(level=logging.INFO, force=True)
    serve(build_argparser().parse_args())


if __name__ == "__main__":
    main()
