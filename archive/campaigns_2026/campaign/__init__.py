"""Student-HP + cross-init campaign: plan constants, selection, and driver support.

Domain modules (`plan`, `pairs`, `selection`, `config_gen`, `state`) are pure and
torch-free; disk / jobq access lives in `driver_io` and the campaign pipeline.
"""
