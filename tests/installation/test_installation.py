import hashlib
import subprocess
import sys

from gev.data.suites import MANIFEST_HASHES, load_manifest, manifest_digest
from gev.infrastructure.pathlookup import reference_bytes


def test_bundled_manifests_are_byte_verified():
    for suite, expected in MANIFEST_HASHES.items():
        assert hashlib.sha256(reference_bytes("suites", suite, "manifest.json")).hexdigest() == expected
        assert manifest_digest(suite) == expected
        assert load_manifest(suite)["files"]


def test_night2_manifest_is_bundled():
    manifest = reference_bytes("night2", "manifest.json")
    assert b'"seed": "night2-20260920"' in manifest


def test_python_module_entrypoint_smoke():
    result = subprocess.run([sys.executable, "-m", "gev", "--help"],
                            capture_output=True, text=True, check=True)

    assert "usage:" in result.stdout.lower()
    assert "study" in result.stdout
