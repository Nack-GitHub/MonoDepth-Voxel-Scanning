"""roomscan_web — thin HTTP front for the roomscan pipeline (Phase 6, after the paper).

Separate package on purpose: it imports `roomscan.pipeline` and nothing in `roomscan`
imports it back, so the experiment code never grows a web dependency.

    uvicorn --factory roomscan_web.app:create_app   # or: make web
"""
