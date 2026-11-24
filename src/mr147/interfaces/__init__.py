"""Contracts with the other parts of the robot.

    vision.py   what the vision team sends us, and how we ask for a better look
    cue_arm.py  what a shot is, and how we ask the cue arm to play it

The real hardware and the simulated hardware (mr147.simulated_hardware) both
implement these, so the rest of the code never knows which one it's talking to.
"""
