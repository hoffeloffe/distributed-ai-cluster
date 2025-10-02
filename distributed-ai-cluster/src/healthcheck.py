#!/usr/bin/env python3
"""Container health check endpoint."""

import asyncio
import logging
import os
from contextlib import suppress

logger = logging.getLogger(__name__)
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))

async def _check_master() -> bool:
    """Placeholder master health verification."""
    # In a future iteration we might call an HTTP endpoint.
    await asyncio.sleep(0)
    return True

async def _check_worker() -> bool:
    """Placeholder worker health verification."""
    await asyncio.sleep(0)
    return True

async def _main() -> None:
    role = os.getenv("NODE_ROLE", "master").lower()
    if role == "master":
        healthy = await _check_master()
    elif role == "worker":
        healthy = await _check_worker()
    else:
        logger.warning("Unknown NODE_ROLE '%s' for healthcheck", role)
        healthy = True

    if not healthy:
        raise SystemExit(1)

if __name__ == "__main__":
    try:
        asyncio.run(_main())
    except SystemExit as exc:
        raise SystemExit(exc.code)
