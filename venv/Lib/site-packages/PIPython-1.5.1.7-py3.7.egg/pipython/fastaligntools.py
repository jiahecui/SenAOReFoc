#!/usr/bin/python
# -*- coding: utf-8 -*-
"""Tools for using the fast alignment routines of a PI device."""

from __future__ import print_function
from time import sleep, time

from pipython.pitools import waitonready

__signature__ = 0x9988f2fea4a76dea29a96f364c84e0a9


class TargetType(object):  # Too few public methods pylint: disable=R0903
    """Enum for TargetType."""
    name = {
        0: 'Sinusoidal', 1: 'Spiral constant frequency', 2: 'Spiral constant path velocity',
    }
    sinusoidal = 0
    spiral_constant_frequency = 1
    spiral_constant_path_velocity = 2


class EstimationMethod(object):  # Too few public methods pylint: disable=R0903
    """Enum for EstimationMethod."""
    name = {
        0: 'Maximum value', 1: 'Gaussian ls fit', 2: 'Center of gravity',
    }
    maximum_value = 0
    gaussian_ls_fit = 1
    center_of_gravity = 2


class StopOption(object):  # Too few public methods pylint: disable=R0903
    """Enum for StopOption."""
    name = {
        0: 'Move to maximum intensity', 1: 'Stay at end of scan', 2: 'Move to start of scan', 3: 'Stop at threshold',
        4: 'Continuous until threshold',
    }
    move_to_maximum_intensity = 0
    stay_at_end_of_scan = 1
    move_to_start_of_scan = 2
    stop_at_threshold = 3
    continuous_until_threshold = 4


class ResultID(object):  # Too few public methods pylint: disable=R0903
    """Enum for ResultID."""
    name = {
        1: 'Success', 2: 'Maximum value', 3: 'Maximum position', 4: 'Routine definition', 5: 'Scan time',
        6: 'Reason of abort', 7: 'Radius', 8: 'Number of direction changes', 9: 'Gradient length',
    }
    success = 1
    max_value = 2
    max_position = 3
    routine_definition = 4
    scan_time = 5
    reason_of_abort = 6
    radius = 7
    number_direction_changes = 8
    gradient_length = 9


# Too many arguments (6/5) pylint: disable=R0913
def waitonalign(pidevice, routines=None, timeout=60, predelay=0, postdelay=0, polldelay=0.1):
    """Wait until all fast align 'routines' are finished, i.e. do not run or are paused.
    @type pidevice : pipython.gcscommands.GCSCommands
    @param routines : Name of the routines as int, string or list, or None to wait for all routines.
    @param timeout : Timeout in seconds as float, defaults to 60 seconds.
    @param predelay : Time in seconds as float until querying any state from controller.
    @param postdelay : Additional delay time in seconds as float after reaching desired state.
    @param polldelay : Delay time between polls in seconds as float.
    """
    waitonready(pidevice, timeout, predelay)
    maxtime = time() + timeout
    while 2 in list(pidevice.qFRP(routines).values()):
        if time() > maxtime:
            raise SystemError('waitonalign() timed out after %.1f seconds' % timeout)
        sleep(polldelay)
    sleep(postdelay)
