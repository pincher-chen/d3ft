"""Opt-in validated corrected-phase trial-forward implementation."""
import runpy,sys
from production_candidate import Installer
if len(sys.argv)<2:raise SystemExit('Usage: optimized_driver.py DRIVER.py [arguments]')
driver=sys.argv[1];sys.argv=[driver,*sys.argv[2:]]
installer=Installer();installer.install()
try:runpy.run_path(driver,run_name='__main__')
finally:installer.restore()
