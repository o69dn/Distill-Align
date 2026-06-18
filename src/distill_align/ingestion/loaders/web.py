"""
Web page loader for fetching content from URLs.

Uses httpx to download web pages and beautifulsoup4 to extract content.

Security: SSRF-protected with URL validation, IP blocking, redirect following limits,
and response body size caps.
"""

import asyncio
import ipaddress
import socket
from typing import Any
from urllib.parse import urljoin, urlparse

from ...core.exceptions import LoaderError, SSRFError
from ...core.schemas import DataChunk, SourceMetadata

# ---------------------------------------------------------------------------
# SSRF protection constants
# ---------------------------------------------------------------------------

# Blocked IP networks (private, loopback, link-local, multicast, reserved)
_BLOCKED_NETWORKS = [
    ipaddress.ip_network("0.0.0.0/8"),            # Current network
    ipaddress.ip_network("10.0.0.0/8"),           # RFC 1918 private
    ipaddress.ip_network("100.64.0.0/10"),        # Carrier-grade NAT
    ipaddress.ip_network("127.0.0.0/8"),          # Loopback
    ipaddress.ip_network("169.254.0.0/16"),       # Link-local (cloud metadata)
    ipaddress.ip_network("172.16.0.0/12"),        # RFC 1918 private
    ipaddress.ip_network("192.0.0.0/24"),         # IETF protocol assignments
    ipaddress.ip_network("192.0.2.0/24"),         # Documentation / TEST-NET-1
    ipaddress.ip_network("192.168.0.0/16"),       # RFC 1918 private
    ipaddress.ip_network("198.18.0.0/15"),        # Benchmarking
    ipaddress.ip_network("198.51.100.0/24"),      # Documentation / TEST-NET-2
    ipaddress.ip_network("203.0.113.0/24"),       # Documentation / TEST-NET-3
    ipaddress.ip_network("224.0.0.0/4"),          # Multicast
    ipaddress.ip_network("240.0.0.0/4"),          # Reserved
    ipaddress.ip_network("255.255.255.255/32"),   # Limited broadcast
    ipaddress.ip_network("::1/128"),              # IPv6 loopback
    ipaddress.ip_network("fc00::/7"),             # IPv6 unique-local
    ipaddress.ip_network("fe80::/10"),            # IPv6 link-local
    ipaddress.ip_network("ff00::/8"),             # IPv6 multicast
]

MAX_REDIRECTS = 5
MAX_BODY_BYTES = 10 * 1024 * 1024  # 10 MB maximum download


# ---------------------------------------------------------------------------
# URL / IP validation helpers
# ---------------------------------------------------------------------------


async def _resolve_and_check(hostname: str, port: int = 443) -> None:
    """Resolve *hostname* to IP addresses and verify none are blocked.

    Raises:
        SSRFError: if the hostname cannot be resolved or resolves to a blocked
            network (private, loopback, link-local, etc.).
    """
    loop = asyncio.get_event_loop()
    try:
        addrinfo = await loop.getaddrinfo(hostname, port, type=socket.SOCK_STREAM)
    except socket.gaierror as exc:
        raise SSRFError(f"Cannot resolve hostname '{hostname}': {exc}") from exc

    checked: set[str] = set()
    for family, _type, _proto, _canonname, sockaddr in addrinfo:
        ip_str = sockaddr[0]
        if ip_str in checked:
            continue
        checked.add(ip_str)
        try:
            ip = ipaddress.ip_address(ip_str)
        except ValueError:
            continue

        for net in _BLOCKED_NETWORKS:
            if ip in net:
                raise SSRFError(
                    f"URL resolves to blocked address {ip_str} "
                    f"(network {net.with_prefixlen})"
                )


def _validate_url(url: str) -> str:
    """Validate and normalise a URL before fetching.

    Checks:
        * Scheme is ``http`` or ``https`` only.
        * Host is present.

    Returns:
        The (possibly normalised) URL string.

    Raises:
        SSRFError: on invalid scheme or missing host.
    """
    parsed = urlparse(url)

    if parsed.scheme not in ("http", "https"):
        raise SSRFError(
            f"URL scheme '{parsed.scheme}' is not allowed (use http or https)"
        )

    if not parsed.netloc:
        raise SSRFError("URL must include a host (e.g. https://example.com)")

    return url


async def _validate_and_resolve(url: str) -> None:
    """Convenience: validate *url* then resolve its host against blocked nets."""
    _validate_url(url)
    parsed = urlparse(url)
    hostname: str = parsed.hostname or parsed.netloc
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    await _resolve_and_check(hostname, port)


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------


class WebLoader:
    """Loader for web pages/URLs.

    Unlike file-based loaders, this loader fetches content from URLs over HTTP.
    It does **not** extend ``BaseLoader`` because its interface is URL-based and
    async.

    **Security**

    *   Only ``http`` / ``https`` URLs are accepted.
    *   Every URL (including redirect targets) is DNS-resolved and checked
        against a blocklist of private / loopback / link-local / metadata
        address ranges.
    *   Redirects are followed manually (max ``MAX_REDIRECTS`` hops).
    *   Response body is streamed and capped at ``MAX_BODY_BYTES`` (10 MB).
    """

    SUPPORTED_EXTENSIONS: list[str] = []  # Not file-based

    async def load_url(self, url: str, **kwargs: Any) -> list[DataChunk]:
        """Load content from a *url*.

        Parameters
        ----------
        url:
            URL to fetch.
        **kwargs:
            * ``selector`` — CSS selector to extract specific content.
            * ``max_length`` — maximum *output* content length (default 100 000).
            * ``max_body_bytes`` — maximum *download* size (default 10 MB).

        Returns
        -------
        list[DataChunk]
            Extracted content as chunks (typically a single chunk).
        """
        import httpx
        from bs4 import BeautifulSoup

        selector = kwargs.get("selector")
        max_length = kwargs.get("max_length", 100_000)
        max_body_bytes = kwargs.get("max_body_bytes", MAX_BODY_BYTES)

        # -- Validate the initial URL -------------------------------------------
        await _validate_and_resolve(url)

        current_url: str = url
        redirects_left = MAX_REDIRECTS

        async with httpx.AsyncClient(
            timeout=30.0,
            follow_redirects=False,  # we follow manually with validation
        ) as client:
            # -- Manual redirect following ---------------------------------------
            while True:
                response = await client.get(
                    current_url,
                    headers={"User-Agent": "Distill-Align/1.0 WebLoader"},
                )

                if response.status_code in (301, 302, 303, 307, 308):
                    redirect_target = response.headers.get("location", "")
                    if not redirect_target:
                        raise LoaderError("Redirect response missing Location header")

                    redirect_target = urljoin(current_url, redirect_target)

                    # Validate every redirect hop
                    if redirects_left <= 0:
                        raise LoaderError(
                            f"Too many redirects (max {MAX_REDIRECTS})"
                        )
                    await _validate_and_resolve(redirect_target)

                    current_url = redirect_target
                    redirects_left -= 1
                    continue

                response.raise_for_status()
                break

            # -- Stream body with size cap ------------------------------------
            chunks: list[bytes] = []
            total = 0
            truncated = False
            async for chunk in response.aiter_bytes():
                remaining = max_body_bytes - total
                if remaining <= 0:
                    truncated = True
                    break
                chunks.append(chunk[:remaining])
                total += len(chunks[-1])

            raw = b"".join(chunks)
            text = raw.decode("utf-8", errors="replace")
            if truncated:
                text += "\n\n[Content truncated due to download size limit...]"

        # -- Parse HTML ----------------------------------------------------------
        soup = BeautifulSoup(text, "html.parser")

        # Remove script / style / navigation elements
        for element in soup(["script", "style", "nav", "footer", "header"]):
            element.decompose()

        if selector:
            elements = soup.select(selector)
            content = "\n\n".join(el.get_text(strip=True) for el in elements)
        else:
            main = soup.find("main") or soup.find("article") or soup.find("body")
            content = (
                main.get_text(strip=True, separator="\n")
                if main
                else soup.get_text(strip=True, separator="\n")
            )

        # Truncate output content if requested
        if len(content) > max_length:
            content = content[:max_length] + "\n\n[Content truncated...]"

        title = (
            soup.title.string.strip()
            if soup.title and soup.title.string
            else url
        )

        metadata = SourceMetadata(
            source_type="text",
            file_path=url,
            file_name=url.split("/")[-1] or "webpage",
            title=title,
            custom_tags={"url": url, "selector": selector},
        )

        return [DataChunk(content=content, metadata=metadata)]

    def load_sync(self, url: str, **kwargs: Any) -> list[DataChunk]:
        """Synchronous wrapper around :meth:`load_url`.

        Parameters
        ----------
        url:
            URL to fetch.
        **kwargs:
            Forwarded to :meth:`load_url`.

        Returns
        -------
        list[DataChunk]
        """
        return asyncio.run(self.load_url(url, **kwargs))

    @staticmethod
    def get_supported_extensions() -> list[str]:
        """Return empty list since web loader doesn't use file extensions."""
        return []
