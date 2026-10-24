import socket
import unittest

from url_safety import UnsafeURL, validate_public_http_url


def resolver_for(*addresses: str):
    def resolve(hostname, port, *, type):
        del hostname, type
        answers = []
        for address in addresses:
            if ":" in address:
                answers.append(
                    (socket.AF_INET6, socket.SOCK_STREAM, 6, "", (address, port, 0, 0))
                )
            else:
                answers.append(
                    (socket.AF_INET, socket.SOCK_STREAM, 6, "", (address, port))
                )
        return answers

    return resolve


class PublicURLValidationTests(unittest.TestCase):
    def test_accepts_public_https_destination(self):
        url = "https://example.com/report?id=42"

        self.assertEqual(
            validate_public_http_url(url, resolver=resolver_for("93.184.216.34")),
            url,
        )

    def test_rejects_loopback_and_private_ipv4_destinations(self):
        for url, address in (
            ("http://127.0.0.1/admin", "127.0.0.1"),
            ("http://internal.example/", "10.0.0.8"),
            ("http://metadata.example/", "169.254.169.254"),
        ):
            with self.subTest(url=url), self.assertRaises(UnsafeURL):
                validate_public_http_url(url, resolver=resolver_for(address))

    def test_rejects_ipv6_loopback(self):
        with self.assertRaises(UnsafeURL):
            validate_public_http_url(
                "http://[::1]/admin", resolver=resolver_for("::1")
            )

    def test_rejects_localhost_without_resolving_it(self):
        def unexpected_resolver(*args, **kwargs):
            raise AssertionError(f"resolver called with {args!r} {kwargs!r}")

        with self.assertRaisesRegex(UnsafeURL, "local destinations"):
            validate_public_http_url(
                "http://service.localhost/", resolver=unexpected_resolver
            )

    def test_rejects_embedded_credentials(self):
        with self.assertRaisesRegex(UnsafeURL, "credentials"):
            validate_public_http_url(
                "https://user:secret@example.com/",
                resolver=resolver_for("93.184.216.34"),
            )

    def test_rejects_non_http_scheme(self):
        with self.assertRaisesRegex(UnsafeURL, "HTTP and HTTPS"):
            validate_public_http_url(
                "file:///etc/passwd", resolver=resolver_for("93.184.216.34")
            )

    def test_rejects_mixed_public_and_private_dns_answers(self):
        with self.assertRaises(UnsafeURL):
            validate_public_http_url(
                "https://mixed.example/",
                resolver=resolver_for("93.184.216.34", "192.168.1.20"),
            )

    def test_fails_closed_when_dns_resolution_fails(self):
        def failing_resolver(*args, **kwargs):
            raise socket.gaierror("temporary failure")

        with self.assertRaisesRegex(UnsafeURL, "could not be resolved"):
            validate_public_http_url(
                "https://unresolved.example/", resolver=failing_resolver
            )


if __name__ == "__main__":
    unittest.main()
