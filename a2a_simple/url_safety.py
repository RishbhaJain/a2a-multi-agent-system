"""Network-boundary validation for browser-agent URLs."""

from __future__ import annotations

import ipaddress
import socket
from collections.abc import Callable
from urllib.parse import urlsplit


class UnsafeURL(ValueError):
    """Raised when a URL is not safe for the browser agent to request."""


def validate_public_http_url(
    url: str,
    *,
    resolver: Callable[..., list[tuple]] = socket.getaddrinfo,
) -> str:
    """Return *url* only when every resolved address is globally routable.

    The browser agent is intentionally limited to public HTTP(S) destinations.
    Resolution failures are denied so that a validation error cannot become an
    accidental bypass.
    """

    try:
        parsed = urlsplit(url)
        port = parsed.port
    except ValueError as exc:
        raise UnsafeURL("URL contains an invalid port") from exc

    if parsed.scheme.lower() not in {"http", "https"}:
        raise UnsafeURL("only HTTP and HTTPS URLs are allowed")
    if not parsed.hostname:
        raise UnsafeURL("URL must include a hostname")
    if parsed.username is not None or parsed.password is not None:
        raise UnsafeURL("embedded credentials are not allowed")

    hostname = parsed.hostname.rstrip(".").lower()
    if hostname == "localhost" or hostname.endswith(".localhost"):
        raise UnsafeURL("local destinations are not allowed")

    destination_port = port or (443 if parsed.scheme.lower() == "https" else 80)
    try:
        answers = resolver(hostname, destination_port, type=socket.SOCK_STREAM)
    except OSError as exc:
        raise UnsafeURL("hostname could not be resolved") from exc

    addresses: set[ipaddress.IPv4Address | ipaddress.IPv6Address] = set()
    for family, _socket_type, _protocol, _canonical_name, sockaddr in answers:
        if family not in {socket.AF_INET, socket.AF_INET6}:
            continue
        try:
            addresses.add(ipaddress.ip_address(sockaddr[0]))
        except ValueError as exc:
            raise UnsafeURL("resolver returned an invalid address") from exc

    if not addresses:
        raise UnsafeURL("hostname did not resolve to an IP address")
    if any(not address.is_global for address in addresses):
        raise UnsafeURL("local, private, and reserved destinations are not allowed")

    return url
