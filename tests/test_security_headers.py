"""Response hardening.

1.x set no security headers and relied on inline scripts and inline event
handlers, so a restrictive policy was not possible.
"""

import re

import pytest

from cropcraft.middleware import SECURITY_HEADERS

PAGES = ["/", "/crop", "/fertilizer", "/pesticide"]


@pytest.mark.parametrize("path", PAGES)
@pytest.mark.parametrize("header", sorted(SECURITY_HEADERS))
def test_header_is_set(client, path, header):
    assert client.get(path).headers[header] == SECURITY_HEADERS[header]


def test_csp_forbids_inline_script(client):
    policy = client.get("/").headers["Content-Security-Policy"]
    assert "script-src 'self'" in policy
    assert "unsafe-inline" not in policy
    assert "frame-ancestors 'none'" in policy


@pytest.mark.parametrize("path", PAGES)
def test_pages_have_no_inline_script(client, path):
    """Every page must survive the CSP above."""
    body = client.get(path).text
    assert not re.search(r"<script(?![^>]*\ssrc=)[^>]*>", body), "inline <script> block"
    assert not re.search(r"\son(?:error|click|submit|load)\s*=", body), "inline event handler"


@pytest.mark.parametrize("path", PAGES)
def test_scripts_are_same_origin(client, path):
    body = client.get(path).text
    for src in re.findall(r'<script[^>]*\ssrc="([^"]+)"', body):
        assert not src.startswith(("http://", "https://")) or src.startswith(str(client.base_url))


@pytest.mark.parametrize("path", ["/docs", "/redoc"])
def test_docs_get_a_scoped_policy(client, path):
    """Swagger UI and ReDoc load from a CDN and self-bootstrap with inline
    script, so the strict app policy would blank them out."""
    from cropcraft.middleware import DOCS_CONTENT_SECURITY_POLICY

    response = client.get(path)
    assert response.status_code == 200
    assert response.headers["Content-Security-Policy"] == DOCS_CONTENT_SECURITY_POLICY
    assert "cdn.jsdelivr.net" in response.headers["Content-Security-Policy"]


def test_docs_exception_does_not_leak_into_app_pages(client):
    """The relaxed policy must apply only to the documentation routes."""
    for path in PAGES:
        policy = client.get(path).headers["Content-Security-Policy"]
        assert "unsafe-inline" not in policy
        assert "jsdelivr" not in policy
