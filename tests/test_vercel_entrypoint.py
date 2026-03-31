import unittest

from fastapi import FastAPI

from app import app


class VercelEntrypointTests(unittest.TestCase):
    def test_root_entrypoint_exports_fastapi_app(self) -> None:
        self.assertIsInstance(app, FastAPI)


if __name__ == "__main__":
    unittest.main()
