#!/usr/bin/env python
#
# For basic launch, run:
# python ngrok.py
import os
from pathlib import Path
import sys
import asyncio, ngrok
import click

from streamlit.web.bootstrap import run

NGROK_PORT = int(os.environ.get("NGROK_PORT", "8501"))
NGROK_DOMAIN = os.environ.get("NGROK_DOMAIN", None)


async def setup_tunnel():
    listen = f"localhost:{NGROK_PORT}"
    session = await ngrok.NgrokSessionBuilder().authtoken_from_env().connect()
    if NGROK_DOMAIN:
        tunnel = await (
            session.http_endpoint()
            .domain(NGROK_DOMAIN)
            .listen()  # .domain('<name>.ngrok.app') # if on a paid plan, set a custom static domain
        )
    else:
        tunnel = await (
            session.http_endpoint().listen()  # .domain('<name>.ngrok.app') # if on a paid plan, set a custom static domain
        )
    click.secho(
        f"Forwarding to {listen} from ingress url: {tunnel.url()}",
        fg="green",
        bold=True,
    )
    tunnel.forward(listen)  # forward_tcp


try:
    # Check if asyncio loop is already running. If so, piggyback on it to run the ngrok tunnel.
    running_loop = asyncio.get_running_loop()
    running_loop.create_task(setup_tunnel())
except RuntimeError:
    # No existing loop is running, so we can run the ngrok tunnel on a new loop.
    asyncio.run(setup_tunnel())

# forward sys.argv
run(str(Path(__file__).parent / "app.py"), command_line=None, args=sys.argv[2:], flag_options={})
