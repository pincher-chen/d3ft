"""User-approved independent GPU6 cheap pilot, alongside the E3 FD wave."""
import e7_ct_pilot as pilot
pilot.priority_pending=lambda: None
pilot.run('diamond','cheap',[6],10800,pilot.initial('diamond'))
pilot.report()
