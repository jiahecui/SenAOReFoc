#!/usr/bin/python
# -*- coding: utf-8 -*-
"""Collection of helpers for handling PI device parameters."""

from pipython import GCSError, gcserror

__signature__ = 0x81610efb9fe04361b229ce8a0dbcf688


def applyconfig(pidevice, axis, config):
    """Try to apply 'config' for 'axis' by applyconfig() or CST() function.
    @type pidevice : pipython.gcscommands.GCSDevice
    @param axis: Single axis as string convertible.
    @param config: Name of a configuration existing in PIStages database as string.
    @return : Warning as string or empty string on success.
    """
    try:
        pidevice.dll.applyconfig(items='axis %s' % axis, config=config)
    except AttributeError:  # function not found in DLL
        if not pidevice.HasCST():
            return 'CST command is not supported'
        pidevice.CST(axis, config)
    except GCSError as exc:
        if exc == gcserror.E_10013_PI_PARAMETER_DB_AND_HPA_MISMATCH_LOOSE:
            del pidevice.axes
            return '%s\n%s' % (exc, pidevice.dll.warning.rstrip())
        else:
            raise
    del pidevice.axes
    return ''
