"""Entrypoint for running the Gradio WebUI."""

from __future__ import annotations

import logging

from app.ui.layout import build_demo
from app.config import get_config


def main() -> None:
    """Start the Gradio application."""
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] %(name)s: %(message)s",
    )
    cfg = get_config()

    logging.info("Model will be loaded on first request")

    demo = build_demo()

    demo.launch(
        server_name=cfg.host,
        server_port=cfg.port,
        share=cfg.share,
    )


if __name__ == "__main__":
    main()
