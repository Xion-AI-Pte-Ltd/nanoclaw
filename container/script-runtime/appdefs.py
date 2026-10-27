"""Existing Counto client convention for approved scripts.

Authentication is configured by the deployment through the shared library's
existing environment contract. No session forwarding or authentication shim.
"""
from counto_api_client import CountoCookieSession

sync_requests = CountoCookieSession()
