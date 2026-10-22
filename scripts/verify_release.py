#!/usr/bin/env python3
"""Current branch verification; use the v1.0 tag for its frozen release verifier."""
import runpy
from pathlib import Path
runpy.run_path(str(Path(__file__).with_name('verify_response_release.py')),run_name='__main__')
