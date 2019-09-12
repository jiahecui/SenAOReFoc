#!/usr/bin/python
# -*- coding: utf-8 -*-
"""Provide GCS functions to control a PI device."""
# Trailing newlines pylint: disable=C0305

from collections import OrderedDict
from logging import debug, warning
import platform
import sys

import pipython

# Invalid class name "basestring"  pylint: disable=C0103
# Redefining built-in 'basestring' pylint: disable=W0622
try:
    basestring
except NameError:
    basestring = str

try:
    unicode
except NameError:
    unicode = str

__signature__ = 0x6c420665abc5bd61f550245fe3c05bb2

GCS1DEVICES = ('C-843', 'C-702.00', 'C-880', 'C-848', 'E-621', 'E-625', 'E-665', 'E-816', 'E-516',
               'C-663.10', 'C-863.10', 'MERCURY', 'HEXAPOD', 'TRIPOD', 'E-710', 'F-206', 'E-761')

GCSFUNCTIONS = {
    'E-816': [
        'qERR', 'qI2C', 'qIDN', 'StopAll', 'BDR', 'qBDR', 'AVG', 'qAVG', 'SCH', 'qSCH', 'qSAI', 'MAC_BEG', 'MAC_START',
        'MAC_NSTART', 'MAC_DEL', 'MAC_DEF', 'MAC_END', 'MAC_qDEF', 'MAC_qFREE', 'qMAC', 'IsRunningMacro', 'DEL', 'WPA',
        'RST', 'qHLP', 'DCO', 'qDCO', 'MOV', 'MVR', 'qMOV', 'SVA', 'SVR', 'qSVA', 'MVT', 'qMVT', 'qDIP', 'qPOS', 'qVOL',
        'qOVF', 'qONT', 'SVO', 'qSVO', 'SWT', 'qSWT', 'WTO', 'SPA', 'qSPA',
    ],
    'HEXAPOD': [
        'qIDN', 'AAP', 'qCST', 'qECO', 'qERR', 'FAA', 'FAM', 'FAS', 'DMOV', 'DRV', 'FIO', 'FSA', 'FSC', 'FSM', 'FSN',
        'GetPosStatus', 'GetScanResult', 'HasPosChanged', 'INI', 'IsMoving', 'IsRecordingMacro', 'IsScanning',
        'MAC_BEG', 'MAC_DEL', 'MAC_END', 'MAC_qERR', 'MAC_START', 'MOV', 'MWG', 'NAV', 'NLM', 'PLM', 'qCST', 'qDRR',
        'qFSN', 'qHLP', 'qMAC', 'qNAV', 'qNLM', 'qPLM', 'qPOS', 'qSAI', 'qSAI_ALL', 'qSCT', 'qSGA', 'qSPA', 'qSPI',
        'qSSL', 'qSST', 'qSVO', 'qTAV', 'qVEL', 'qVER', 'SCT', 'SGA', 'SPI', 'SSL', 'SST', 'STP', 'SVO', 'TAV', 'VEL',
        'VMO',
    ],
    'TRIPOD': [
        'qIDN', 'AAP', 'qCST', 'qECO', 'qERR', 'FAA', 'FAM', 'FAS', 'DMOV', 'DRV', 'FIO', 'FSA', 'FSC', 'FSM', 'FSN',
        'GetPosStatus', 'GetScanResult', 'HasPosChanged', 'INI', 'IsMoving', 'IsRecordingMacro', 'IsScanning',
        'MAC_BEG', 'MAC_DEL', 'MAC_END', 'MAC_qERR', 'MAC_START', 'MOV', 'MWG', 'NAV', 'NLM', 'PLM', 'qCST', 'qDRR',
        'qFSN', 'qHLP', 'qMAC', 'qNAV', 'qNLM', 'qPLM', 'qPOS', 'qSAI', 'qSAI_ALL', 'qSCT', 'qSGA', 'qSPA', 'qSPI',
        'qSSL', 'qSST', 'qSVO', 'qTAV', 'qVEL', 'qVER', 'SCT', 'SGA', 'SPI', 'SSL', 'SST', 'STP', 'SVO', 'TAV', 'VEL',
        'VMO',
    ],
    'C-702.00': [
        'IsMoving', 'HasPosChanged', 'IsControllerReady', 'IsRunningMacro', 'StopAll', 'SystemAbort', 'q*IDN', 'ACC',
        'qACC', 'CCL', 'qCCL', 'CLR', 'CLS', 'CST', 'qCST', 'qCSV', 'CTO', 'qCTO', 'DEL', 'DFF', 'qDFF', 'DFH', 'qDFH',
        'DIO', 'qDIO', 'DRC', 'qDRC', 'qDRR', 'DRT', 'qDRT', 'DSP', 'qDSP', 'qECO', 'qERR', 'GOH', 'qHDR', 'HID',
        'qHID', 'qHLP', 'HLT', 'IFC', 'qIFC', 'IFS', 'qIFS', 'INI', 'ITD', 'qLIM', 'MAC', 'qMAC', 'MNL', 'MOV', 'qMOV',
        'MPL', 'MSG', 'MVR', 'MVS', 'qMVS', 'NLM', 'qNLM', 'qONT', 'PLM', 'qPLM', 'POS', 'qPOS', 'RBT', 'REF', 'qREF',
        'RON', 'qRON', 'RST', 'RTR', 'qRTR', 'SAI', 'qSAI', 'SAV', 'SCA', 'qSCA', 'SMO', 'qSMO', 'SPA', 'qSPA', 'SSL',
        'qSSL', 'qSSN', 'SSP', 'SST', 'qSST', 'qSTA', 'STP', 'SVO', 'qSVO', 'qTAC', 'qTCV', 'qTIM', 'qTIO', 'qTMN',
        'qTMX', 'qTNR', 'TRO', 'qTRO', 'qTSP', 'qTVI', 'VEL', 'qVEL', 'qVER', 'VMO', 'qVST', 'WAA', 'WAI', 'WAV',
        'qWAV', 'WGO', 'qWGO', 'WPA', 'WSL', 'qWSL',
    ]
}


def gethexstr(paramids):
    """Create string of 'paramids' with hexadecimal numbers.
    @param paramids : List of integers or integer convertibles of base 10 or 16.
    @return : String of 'paramids' with hexadecimal numbers.
    """
    for i, paramid in enumerate(paramids):
        try:
            paramids[i] = int(paramid)
        except ValueError:
            paramids[i] = int(paramid, base=16)
    return '[' + ', '.join(['0x%x' % x for x in paramids]) + ']'


def getsupportedcommands(qhlp, dostrip=True):
    """Parse qHLP answer and return list of available command names.
    @param qhlp : Answer of qHLP() as string.
    @param dostrip : If True strip first and last line from 'qhlp'.
    @return : List of supported command names (not function names).
    """
    qhlp = qhlp.splitlines()
    if dostrip:
        qhlp = qhlp[1:-1]
    cmds = []
    for line in qhlp:
        line = line.upper()
        cmds.append(line.split()[0].strip())
    return cmds


def getsupportedfunctions(qhlp, dostrip=True):
    """Parse qHLP answer and return list of available functions.
    @param qhlp : Answer of qHLP() as string.
    @param dostrip : If True strip first and last line from 'qhlp'.
    @return : List of supported function names (not command names).
    """
    funcnames = {
        'q*IDN': 'qIDN', '#3': 'GetPosStatus', '#5': 'IsMoving', '#6': 'HasPosChanged', '#7': 'IsControllerReady',
        '#8': 'IsRunningMacro', '#9': 'IsGeneratorRunning', '#11': 'GetDynamicMoveBufferSize', '#24': 'StopAll',
        '#4': 'GetStatus', '#27': 'SystemAbort'
    }
    qhlp = qhlp.splitlines()
    if dostrip:
        qhlp = qhlp[1:-1]
    funcs = []
    for line in qhlp:
        line = line.upper()
        funcname = line.split()[0]
        if funcname.endswith('?'):
            funcname = 'q%s' % funcname[:-1]
        if funcname == 'MAC':
            for item in ('BEG', 'DEF', 'DEF?', 'DEL', 'END', 'ERR?', 'NSTART', 'START', 'FREE?', 'STOP'):
                if line.find(item) >= 0:
                    if item.endswith('?'):
                        item = 'q%s' % item[:-1]
                    funcs.append('MAC_%s' % item)
        else:
            funcs.append(funcnames.get(funcname, funcname))
    return funcs


def getitemslist(items, valueconv=None, size=None):
    """Return list of 'items'.
    @param items : Can be None, single item or list of items.
    @param valueconv : Optionally convert each item in 'items' to this type.
    @param size : Optional size to extend 'items' with its last element as integer.
    @return : List of 'items'.
    """
    if isinstance(items, dict):
        raise TypeError('parameter type mismatch: %r' % items)
    if items in (None, '', {}):
        items = []
    items = items if isinstance(items, (list, set, tuple)) else [items]
    items = list(items)  # tuple has no attribute 'extend'
    if size:
        items.extend([items[-1]] * (size - len(items)))
    if valueconv:
        items = [convertvalue(item, valueconv) for item in items]
    return items


def getitemsvaluestuple(items, values, required=True):
    """Convert single values, lists or a dictionary lists of "items" and "values".
    @param items : Single item or list of items or dictionary of {item : value}.
    @param values : Single value or list of values or None if 'items' is a dictionary.
    @param required: If True 'values' must not be empty.
    @return : Tuple ([items], [values]).
    """
    if isinstance(items, dict):
        if values == None:
            values = list(items.values())
            items = list(items.keys())
        else:
            raise TypeError('parameter type mismatch: If <items> is a dictionary <values> must be "None"')

    items = getitemslist(items)
    values = getitemslist(values)
    if required:
        checksize((True, True), items, values)
    else:
        checksize((), items, values)
    return items, values


def getitemsparamsvaluestuple(itemdict, params, values):
    """Convert single values, lists or a dictionary lists of 'itemdict', 'params' and 'values'.
    @param itemdict : Single item or list of items or dictionary of {item : {param : value}}.
    @param params : Single value or list of values or None if 'itemdict' is a dictionary.
    @param values : Single value or list of values or None if 'itemdict' is a dictionary.
    @return : Tuple ([items], [params], [values]) where all three lists have the same length.
    """
    params = getitemslist(params)
    values = getitemslist(values)
    if isinstance(itemdict, dict):
        items = []
        for item in itemdict:
            for param in itemdict[item]:
                items.append(item)
                params.append(param)
                values.append(itemdict[item][param])
    else:
        items = getitemslist(itemdict)
    checksize((True, True, True), items, params, values)
    return items, params, values


def getitemsparamsidstuple(itemdict, params, ids):
    """Convert single values, lists or a dictionary lists of 'itemdict', 'params' and 'ids'.
    @param itemdict : Single item or list of items or dictionary of {[item, params] : [ids]}.
    @param params : Single value or list of values or None if 'itemdict' is a dictionary.
    @param ids : Single value or list of values or None if 'itemdict' is a dictionary.
    @return : Tuple ([items]], [params], [ids]) where all three lists have the same length.
    """
    params = getitemslist(params)
    ids = getitemslist(ids)
    if isinstance(itemdict, dict):
        items = []
        for item in itemdict:
            #           for param in itemdict[item]:
            #               items.append(item)
            #               params.append(param)
            #               ids.append(itemdict[item][param])
            items.append(item[0])
            params.append(item[1])
            for id in itemdict[item]:
                ids.append(id)
    else:
        items = getitemslist(itemdict)
    checksize((True, len(items), len(ids)), items, params, ids)
    return items, params, ids


# 'convertvalue' is too complex. The McCabe rating is 12 pylint: disable=R1260
#  Too many return statements pylint: disable=R0911
def convertvalue(value, totype):
    """Convert 'value' to 'totype'. For bool conversion "1" and "True" is regarded as True.
    @param value : Usually a string that gets converted. Can have whitespaces.
    @param totype : Type to convert to or None to not convert or True for automatic conversion.
    @return : Converted and stripped 'value'.
    """
    if totype is None:
        return value
    if not isinstance(value, (basestring, int, float, bool)):
        raise TypeError('parameter is of unexpected type: %r' % value)
    if isinstance(value, basestring):
        value = value.strip()
    if totype is True:
        return converttonumber(value)
    if totype is bool:
        if value in ['1', 'True']:
            return True
        if value in ['0', 'False']:
            return False
        raise ValueError('unexpected response %r for bool conversion' % value)
    if totype is int:
        try:
            return int(value, base=0)  # proper base is guessed
        except TypeError:
            return int(value)
    if totype is str and isinstance(value, unicode):  # prevent UnicodeEncodeError
        return value
    return totype(value)


def converttonumber(value):
    """Convert 'value' to int (dec or hex but no oct) or float if possible.
    @param value : String (no other types!) that gets converted. No whitespaces.
    @return : Converted 'value' as number or 'value' itself.
    """
    try:
        if value.find('.') < 0:
            if '0x' in value.lower():
                return int(value, base=16)
            return int(value, base=10)
        return float(value)
    except ValueError:
        try:
            return str(value)
        except UnicodeEncodeError:
            return value


def splitanswertolists(answer):
    """Split 'answer' into tuple (items, values) of lists.
    @param answer : String "item1 item2 ... = val1 val2 ...<LF>" or single "value".
    @return : Tuple ([[item1], [item2], ...], [[val1, val2, ...]]) or (None, [[values]]).
    """
    answer = answer.split('\n')
    if '=' in answer[0]:
        itemdim = len(answer[0].split('=')[0].split())
        items = [[] for _ in range(itemdim)]  # do not use [[]] * itemdim
    else:
        return None, [[x.strip() for x in answer]]
    values = []
    for line in answer:
        line = line.strip()
        if not line:
            continue
        itemvals = line.split('=')[0].split()
        for dim in range(itemdim):
            items[dim].append(itemvals[dim])
        values.append('='.join(line.split('=')[1:]).split())
    return items, values


def getdict_oneitem(answer, items, itemconv=None, valueconv=None):
    """Split 'answer' into item/values dict of according types.
    @param answer : String "item = val1 val2 ...<LF>".
    @param items : Items (e.g. axes/channels) as single item, list or None.
    @param itemconv : Conversion function for 'items'.
    @param valueconv : List of conversion functions for values. If there are more values than
    conversion functions the last given conversion function is used for the remaining values.
    @return : Ordered dictionary {item: [value1, value2, ...]} or {item: value}.
    """
    readitems, values = splitanswertolists(answer)
    if items is None:
        if not readitems:
            return {}
        items = readitems[0]
    else:
        items = getitemslist(items)
        itemconv = None
    answerdict = OrderedDict()
    multival = False
    for row, _ in enumerate(items):
        item = convertvalue(items[row], itemconv)
        answerdict[item] = []
        for colnum, value in enumerate(values[row]):
            if value:
                answerdict[item].append(convertvalue(value, valueconv[min(colnum, len(valueconv) - 1)]))
                multival |= colnum > 0
    if not multival:
        for item in answerdict:
            try:
                answerdict[item] = answerdict[item][0]
            except IndexError:
                answerdict[item] = ''
    return answerdict


# 'getdict_twoitems' is too complex. The McCabe rating is 14 pylint: disable=R1260
#  Too many branches (13/12) pylint: disable=R0912
#  Too many local variables (17/15) pylint: disable=R0914
def getdict_twoitems(answer, items1, items2, itemconv, valueconv, convlisttostring=False):
    """Split 'answer' into item/values dict of according types.
    @param answer : String "item1 item2 = val1 val2 ...<LF>" or tuple ([val1], [val2], ...).
    @param items1 : Items (e.g. axes/channels) as single item, list or empty.
    @param items2 : Items (e.g. axes/channels) as single item, list or empty.
    @param itemconv : List of two conversion functions for items. No tuple!
    @param valueconv : List of conversion functions for values. If there are more values than
    @param convlisttostring : force converting vsalue lists to strings
    conversion functions the last given conversion function is used for the remaining values.
    @return : Ordered dictionary {item1: {item2: [value1, value2, ...]}} or
    {item1: {item2: value}}.
    """
    readitems, values = splitanswertolists(answer)
    if not items1:
        if not readitems:
            return {}
        items1 = readitems[0]
    else:
        items1 = getitemslist(items1, size=len(readitems[0]))
        itemconv[0] = None
    if not items2:
        if not readitems:
            return {}
        items2 = readitems[1]
    else:
        items2 = getitemslist(items2)
        itemconv[1] = None
    answerdict = OrderedDict()
    multival = False
    for row, _ in enumerate(items1):
        item1 = convertvalue(items1[row], itemconv[0])
        item2 = convertvalue(items2[row], itemconv[1])
        if item1 not in answerdict:
            answerdict[item1] = OrderedDict()
        answerdict[item1][item2] = []
        for colnum, value in enumerate(values[row]):
            if value:
                answerdict[item1][item2].append(convertvalue(value, valueconv[min(colnum, len(valueconv) - 1)]))
                multival |= colnum > 0

    if not multival or convlisttostring:
        for item1 in answerdict:
            for item2 in answerdict[item1]:
                try:
                    if not convlisttostring:
                        answerdict[item1][item2] = answerdict[item1][item2][0]
                    else:
                        answerdict[item1][item2] = ' '.join(str(e) for e in answerdict[item1][item2])

                except IndexError:
                    answerdict[item1][item2] = ''
    return answerdict


# 'getdict_threeitems' is too complex. The McCabe rating is 14 pylint: disable=R1260
#  Too many branches (13/12) pylint: disable=R0912
#  Too many local variables (17/15) pylint: disable=R0914
def getdict_threeitems(answer, items1, items2, items3, itemconv, valueconv):
    """Split 'answer' into item/values dict of according types.
    @param answer : String "item1 item2 item3= val1...<LF>" or tuple ([val1], [val2], ...).
    @param items1 : Items (e.g. axes/channels) as single item, list or empty.
    @param items2 : Items (e.g. axes/channels) as single item, list or empty.
    @param items3 : Items (e.g. axes/channels) as single item, list or empty.
    @param itemconv : List of two conversion functions for items. No tuple!
    @param valueconv : List of conversion functions for values. If there are more values than
    conversion functions the last given conversion function is used for the remaining values.
    @return : Ordered dictionary {[item1, item2]: [item3, value]}}
    """
    readitems, values = splitanswertolists(answer)
    if not items1:
        if not readitems:
            return {}
        items1 = readitems[0]
    else:
        items1 = getitemslist(items1, size=len(readitems[0]))
        itemconv[0] = None
    if not items2:
        if not readitems:
            return {}
        items2 = readitems[1]
    else:
        items2 = getitemslist(items2, size=len(readitems[1]))
        itemconv[1] = None
    if not items3:
        if not readitems:
            return {}
        items3 = readitems[2]
    else:
        items3 = getitemslist(items3, size=len(readitems[2]))
        itemconv[2] = None

    answerdict = OrderedDict()
    multival = False
    for row, _ in enumerate(items1):
        item1 = convertvalue(items1[row], itemconv[0])
        item2 = convertvalue(items2[row], itemconv[1])
        item3 = convertvalue(items3[row], itemconv[2])

        if (item1, item2) not in answerdict:
            answerdict[(item1, item2)] = OrderedDict()
        answerdict[(item1, item2)][item3] = []
        for colnum, value in enumerate(values[row]):
            if value:
                answerdict[(item1, item2)][item3].append(
                    convertvalue(value, valueconv[min(colnum, len(valueconv) - 1)])
                )

                multival |= colnum > 0
    if not multival:
        for (item1, item2) in answerdict:
            for item3 in answerdict[(item1, item2)]:
                try:
                    answerdict[(item1, item2)][item3] = answerdict[(item1, item2)][item3][0]
                except IndexError:
                    answerdict[(item1, item2)][item3] = ''
    return answerdict


def splitparams(answer, separator):
    """Split 'answer' into list of strings according to GCS1/GCS2 or 'separator'.
    @param answer : String to split.
    @param separator : True for GCS2, False for GCS1 or separator as string.
    @return : List of strings with removed leading and trailing whitespaces.
    """
    if not answer:
        return []
    answer = answer.strip()
    if isinstance(separator, basestring):
        answer = answer.split(separator)
    else:
        if separator:  # GCS2
            answer = answer.split()
        else:  # GCS1
            answer = list(answer)
    return [x.strip() for x in answer]


def getgcsheader(headerstr):
    """Split textual GCS header to key/value pairs.
    @param headerstr : GCS header with lines "# key = value <LF>" as string.
    @return : Ordered dictionary of header items.
    """
    header = OrderedDict()
    for line in headerstr.split('\n'):
        line = line.lstrip('#').strip()
        items = line.split('=')
        if len(items) == 2:
            header[items[0].strip()] = items[1].strip()
    for key in header:
        header[key] = converttonumber(header[key])
    return header


def getbitcodeditems(value, allitems=None, items=None):
    """Return boolean dictionary of 'items' according to corresponding bit in 'value'.
    If 'allitems' is None the number of bits in 'value' is used.
    @param value : Integer of the bit mask.
    @param allitems : Single item or list or None.
    Length must not match the bit width of 'value'.
    @param items : Item or list or None. Only these items are returned.
    If None then 'allitems' are returned.
    @return : Ordered dictionary {items: value} where value is True or False.
    """
    bits = bin(value)[2:][::-1]  # LSB is left
    if allitems is None:
        items = getitemslist(items)
        maxitem = max([int(x) for x in items]) if items else 0
        allitems = list(range(1, max(len(bits), maxitem) + 1))
    allitems = allitems if isinstance(allitems, (list, set, tuple)) else [allitems]
    answerdict = OrderedDict()
    items = items or allitems
    items = items if isinstance(items, (list, set, tuple)) else [items]
    for i, _ in enumerate(allitems):
        itemfound = None
        for j, _ in enumerate(items):
            if str(allitems[i]) == str(items[j]):
                itemfound = j
                break
        if itemfound is not None:
            try:
                answerdict[items[itemfound]] = bits[i] == '1'
            except IndexError:
                answerdict[items[itemfound]] = False
    return answerdict


def checksize(sizes, *args):
    """Check size of 'args'.
    @param sizes : Desired size of 'args' as list of length <= len(args). Integer for size, True if any size but
    required. If length is smaller than number of 'args' then the remaining args must have the same size.
    @param args : Items to be checked as single items, tuples or lists.
    """
    listsize = None
    for i, arg in enumerate(args):
        arg = getitemslist(arg)
        if i < len(sizes):
            size = sizes[i]
        elif i == len(sizes):
            size = len(arg)
        if size is True:
            if not arg:
                raise TypeError('a required parameter is missing')
            listsize = listsize or len(arg)
            size = listsize
        if size != len(arg):
            raise TypeError('parameter size mismatch: %r' % arg)


def logsysinfo():
    """Log which processor, operating system and python is used, useful for support."""
    debug('Processor: %s', platform.processor())
    debug('OS: %s, Architecture: %s', platform.platform(), platform.machine())
    pyver = '{0.major}.{0.minor}.{0.micro} {0.releaselevel}'.format(sys.version_info)
    debug('Python: %s %s', pyver, platform.architecture()[0])


# Invalid method name pylint: disable=C0103
# Too many lines in module pylint: disable=C0302
# Too many public methods pylint: disable=R0904
# Too many arguments pylint: disable=R0913
class GCSCommands(object):
    """Provide functions for GCS commands and communicate with PI controller."""

    def __init__(self, msgs):
        """Wrapper for PI GCS DLL.
        @type msgs : pipython.gcsmessages.GCSMessages
        """
        debug('create an instance of GCSCommands(msgs=%s)', str(msgs))
        logsysinfo()
        self.__msgs = msgs
        self.__funcs = None
        self.__name = None
        self.__axes = []
        self.__allaxes = []
        self._settings = {'paramconv': {}}

    def __str__(self):
        return 'GCSCommands(msgs=%s)' % str(self.__msgs)

    @staticmethod
    def _int_base(value):
        int_base = 10
        if value.upper().lstrip().find('0X') == 0:
            int_base = 16
        return int_base

    @staticmethod
    def _float(values):
        if isinstance(values, list):
            floatvalues = []
            for value in values:
                floatvalues.append(float(value))
        else:
            floatvalues = float(values)

        return floatvalues

    def _int(self, values):
        if isinstance(values, list):
            intvalues = []
            for value in values:
                intvalues.append(int(value, self._int_base(value)))

        else:
            intvalues = int(values, self._int_base(values))

        return intvalues

    @property
    def logfile(self):
        """Full path to file where to save communication to/from device."""
        return self.__msgs.logfile

    @logfile.setter
    def logfile(self, filepath):
        """Full path to file where to save communication to/from device."""
        self.__msgs.logfile = filepath

    @property
    def connectionid(self):
        """Get ID of current connection as integer."""
        return self.__msgs.connectionid

    def GetID(self):
        """Get ID of current connection as integer.
        DEPRECATED: Use GCSCommands.connectionid instead.
        """
        return self.__msgs.connectionid

    @property
    def devname(self):
        """Return device name from its IDN string."""
        if self.__name is None:
            idn = self.qIDN().upper()
            if 'PI-E816' in idn:
                self.__name = 'E-816'
            elif 'DIGITAL PIEZO CONTROLLER' in idn:
                self.__name = 'E-710'
            else:
                self.__name = idn.split(',')[1].strip()
            debug('GCSCommands.devname: set to %r', self.__name)
        return self.__name

    @devname.setter
    def devname(self, devname):
        """Set device name as string, only for testing."""
        self.__name = str(devname)
        warning('controller name is coerced to %r', self.__name)

    @devname.deleter
    def devname(self):
        """Reset device name."""
        self.__name = None
        debug('GCSCommands.devname: reset')

    @property
    def isgcs2(self):
        """True if connected device is a GCS 2 device."""
        return self.devname not in GCS1DEVICES

    @property
    def axes(self):
        """Get connected axes as list of strings."""
        if not self.__axes:
            self.__axes = self.qSAI()
            debug('GCSCommands.axes: set to %r', self.__axes)
        return self.__axes

    @axes.setter
    def axes(self, axes):
        """Set connected axes to 'axes' as list of strings. For testing only."""
        self.__axes = axes if isinstance(axes, (list, set, tuple)) else [axes]
        warning('GCSCommands.axes: coerced to %r', self.__axes)

    @axes.deleter
    def axes(self):
        """Reset axes and allaxes property."""
        self.__axes = []
        self.__allaxes = []
        debug('GCSCommands.axes: reset')

    @property
    def numaxes(self):
        """Get number of connected axes.
        @return : Number of connected axes as integer.
        """
        return len(self.axes)

    @property
    def allaxes(self):
        """Get all axes names as list of strings."""
        if not self.__allaxes:
            self.__allaxes = self.qSAI_ALL()
            debug('GCSCommands.allaxes: set to %r', self.__allaxes)
        return self.__allaxes

    @allaxes.setter
    def allaxes(self, allaxes):
        """Set all axes to 'allaxes' as list of strings. For testing only."""
        self.__allaxes = allaxes if isinstance(allaxes, (list, set, tuple)) else [allaxes]
        warning('GCSCommands.allaxes: coerced to %r', self.__allaxes)

    @property
    def errcheck(self):
        """Get current error check setting."""
        return self.__msgs.errcheck

    @errcheck.setter
    def errcheck(self, value):
        """Set error check property.
        @param value : True means that after each command the error is queried.
        """
        self.__msgs.errcheck = bool(value)
        debug('GCSCommands.errcheck set to %s', self.__msgs.errcheck)

    def checkerror(self):
        """Query error from controller and raise an GCSError."""
        error = self.qERR()
        if error:
            raise pipython.GCSError(error)

    @property
    def floatformat(self):
        """Get format specifier that formats float arguments into command strings."""
        if 'floatformat' not in self._settings:
            self._settings['floatformat'] = '.12g'
        return self._settings['floatformat']

    @floatformat.setter
    def floatformat(self, value):
        """Set format specifier that formats float arguments into command strings.
        @param value : String, e.g. "f" for fixed point or "e" for scientific notification.
        """
        debug('set float format specifier to %r', value)
        self._settings['floatformat'] = value

    def getcmdstr(self, cmd, *args):
        """Convert 'cmd' and all 'args' into a GCS1 or GCS2 string.
        @param cmd : Command as string.
        @param args : Single items or lists of string convertibles, can have different lengths.
        @return : String of 'cmd' and zipped arguments.
        """
        params = []
        for arg in args:
            params.append(getitemslist(arg))
        items = []
        if params:
            for i in range(max([len(param) for param in params])):
                for param in params:
                    if i < len(param):
                        param = param[i]
                        if isinstance(param, bool):
                            param = '1' if param else '0'
                        elif isinstance(param, float):
                            param = format(param, self.floatformat)
                        items.append('%s ' % str(param))
        cmdstr = '%s %s' % (cmd, ''.join(items))
        return cmdstr.strip()

    def SetErrorCheck(self, value):
        """Set error check property to 'value' and return current value.
        DEPRECATED: Use GCSMessages.errcheck instead.
        @param value : True means that after each command the error is queried.
        @return : Current setting as bool.
        """
        errcheck = self.__msgs.errcheck
        self.__msgs.errcheck = bool(value)
        debug('DEPRECATED -- GCSCommands.SetErrorCheck(value=%r): %r', value, errcheck)
        return errcheck

    @property
    def embederr(self):
        """Get current embed error setting, i.e. if "ERR?" is embedded into a set command."""
        return self.__msgs.embederr

    @embederr.setter
    def embederr(self, value):
        """Set embed error property.
        @param value : True means that "ERR?" is embedded into a set command.
        """
        self.__msgs.embederr = bool(value)
        debug('GCSCommands.embederr set to %s', self.__msgs.embederr)

    @property
    def timeout(self):
        """Get current timeout setting in milliseconds."""
        return self.__msgs.timeout

    @timeout.setter
    def timeout(self, value):
        """Set timeout.
        @param value : Timeout in milliseconds as integer.
        """
        self.__msgs.timeout = int(value)

    def SetTimeout(self, value):
        """Set timeout to 'value' and return current value.
        DEPRECATED: Use GCSMessages.timeout instead.
        @param value : Timeout in milliseconds as integer.
        @return : Current timeout in milliseconds as integer.
        """
        timeout = self.__msgs.timeout
        self.__msgs.timeout = int(value)
        debug('DEPRECATED -- GCSCommands.SetTimeout(value=%r): %r', value, timeout)
        return timeout

    @property
    def funcs(self):
        """Return list of supported GCS functions."""
        if self.__funcs is None:
            if self.devname in GCSFUNCTIONS:
                self.__funcs = GCSFUNCTIONS[self.devname]
            else:
                self.__funcs = getsupportedfunctions(self.qHLP())
        return self.__funcs

    @funcs.deleter
    def funcs(self):
        """Reset list of supported GCS functions."""
        debug('GCSCommands.funcs: reset')
        self.__funcs = None

    def __has(self, funcname):
        """Return True if connected controller supports the command that is called by 'funcname'.
        @param funcname : Case sensitive name of DLL function.
        @return : True if controller supports GCS command according to 'func'.
        """
        hasfunc = funcname in self.funcs
        debug('GCSCommands.Has%s = %s', funcname, hasfunc)
        return hasfunc

    @property
    def bufstate(self):
        """False if no buffered data is available. True if buffered data is ready to use.
        Float value 0..1 indicates read progress. To wait, use "while self.bufstate is not True".
        """
        return self.__msgs.bufstate

    @property
    def bufdata(self):
        """Get buffered data as 2-dimensional list of float values.
        Use "while self.bufstate is not True" and then call self.bufdata to get the data. (see docs)
        """
        return self.__msgs.bufdata

    def GcsCommandset(self, tosend):
        """Send 'tosend' to device, there will not be any check for error.
        @param tosend : String to send to device, with or without trailing linefeed.
        """
        debug('GCSCommands.GcsCommandset(%r)', tosend)
        checksize((1,), tosend)
        errcheck = self.__msgs.errcheck
        self.__msgs.errcheck = False
        self.__msgs.send(tosend)
        self.__msgs.errcheck = errcheck

    def send(self, tosend):
        """Send 'tosend' to device and check for error.
        @param tosend : String to send to device, with or without trailing linefeed.
        """
        debug('GCSCommands.send(%r)', tosend)
        checksize((1,), tosend)
        self.__msgs.send(tosend)

    def ReadGCSCommand(self, tosend):
        """Send 'tosend' to device, read answer, there will not be any check for error.
        @param tosend : String to send to device.
        @return : Device answer as string.
        """
        debug('GCSCommands.ReadGCSCommand(%s)', tosend)
        checksize((1,), tosend)
        errcheck = self.__msgs.errcheck
        self.__msgs.errcheck = False
        answer = self.__msgs.read(tosend)
        self.__msgs.errcheck = errcheck
        debug('GCSCommands.ReadGCSCommand = %r', answer)
        return answer

    def read(self, tosend):
        """Send 'tosend' to device, read answer and check for error.
        @param tosend : String to send to device.
        @return : Device answer as string.
        """
        debug('GCSCommands.read(%s)', tosend)
        checksize((1,), tosend)
        answer = self.__msgs.read(tosend)
        debug('GCSCommands.read = %r', answer)
        return answer

    def read_gcsdata(self, tosend):
        """Send 'tosend' to the device and read GCS data to buffer.
        Use "while self.bufstate is not True" and then call self.bufdata to get the data. (see docs)
        @param tosend : String to send to device.
        @return : Header as ordered dictionary.
        """
        debug('GCSCommands.read_gcsdata(tosend=%r)', tosend)
        checksize((1,), tosend)
        answer = self.__msgs.read(tosend, gcsdata=None)
        answer = getgcsheader(answer)
        debug('GCSCommands.read_gcsdata = %r', answer)
        return answer

    def getparam(self, param, item=1):
        """Try to read 'param' for 'item', return None if 'param' is not available.
        @param param : Single parameter ID as integer.
        @param item : Single item ID as integer, defaults to "1".
        @return : Value returned or None.
        """
        if not self.HasSPA():
            return None
        try:
            value = self.qSPA(item, param)[item][param]
        except pipython.GCSError as exc:
            if exc in (pipython.gcserror.E54_PI_CNTR_UNKNOWN_PARAMETER,
                       pipython.gcserror.E1_PI_CNTR_PARAM_SYNTAX,
                       pipython.gcserror.E_7_COM_TIMEOUT,
                       pipython.gcserror.E_1020_PI_INVALID_SPA_CMD_ID):
                return None
            else:
                raise
        return value

    def paramconv(self, paramdict):
        """Convert values in 'paramdict' to according type in qHPA answer.
        @paramdict: Dictionary of {item: {param: value}}.
        @return: Dictionary of {item: {param: value}}.
        """
        self.initparamconv()

        for item in paramdict:
            for param in paramdict[item]:
                if param in self._settings['paramconv']:
                    paramdict[item][param] = self._settings['paramconv'][param](paramdict[item][param])

        return paramdict


    def initparamconv(self):
        """Initialize paramconv .
        """
        if not self._settings['paramconv']:
            for line in self.qHPA().splitlines():
                if '=' not in line:
                    continue
                paramid = int(line.split('=')[0].strip(), base=16)
                convtype = line.split()[3].strip()
                if convtype.upper() == 'INT':
                    self._settings['paramconv'][paramid] = self._int
                elif convtype.upper() == 'FLOAT':
                    self._settings['paramconv'][paramid] = self._float
                elif convtype.upper() in ('CHAR', 'STRING'):
                    continue
                else:
                    raise KeyError('unknown parameter type %r' % convtype)


    def clearparamconv(self):
        """Clear the stored parameter conversion settings."""
        debug('GCSCommands.clearparamconv()')
        self._settings['paramconv'] = {}


    # GCS FUNCTIONS ### DO NOT MODIFY THIS LINE !!! ###############################################

    def FDG(self, routinename, scanaxis, stepaxis, minlevel=None, aligninputchannel=None, minampl=None, maxampl=None,
            frequency=None, speedfactor=None, maxvelocity=None, maxdirectionchanges=None, speedoffset=None):
        """Define a fast alignment gradient search process.
        Use FRS to start the process and FRR? to read out the results of the process.
        Optional parameters need to be scalar, i.e. not list, tuple, dict.
        @param routinename : Name of the fast alignment routine to define, must be unique and != 0.
        @param scanaxis : Axis that is defined to be the master axis of the scan process.
        @param stepaxis : Axis that is defined to be the second axis of the scan process.
        @param minlevel : If the given minimum level is not met by the algorithm, FRS? reports
        "not successful". If set to 0, process will continually track the maximum signal.
        Defaults to 0.1.
        @param aligninputchannel : Input channel to align, defaults to 1.
        @param minampl : Minimum amplitude of dither sinusoidal signal for circular motion.
        Defaults to 2 microns.
        @param maxampl : Maximum amplitude of dither sinusoidal signal for circular motion.
        Defaults to 4 microns.
        @param frequency : Frequency of dither, defaults to 10 Hz.
        @param speedfactor : Factor that acts as an integral term to adapt the offset of
        the dither, defaults to 10.
        @param maxvelocity : Maximum velocity of the offset.
        @param maxdirectionchanges : Second condition to stop a gradient scan.
        @param speedoffset : Relative speed offset to achieve faster alignment times for
        very small gradients, range is 0..1.
        """
        debug('GCSCommands.FDG(routinename=%r, scanaxis=%r, stepaxis=%r, minlevel=%r, aligninputchannel=%r, '
              'minampl=%r, maxampl=%r, frequency=%r, speedfactor=%r, maxvelocity=%r, maxdirectionchanges=%r, '
              'speedoffset=%r)', routinename, scanaxis, stepaxis, minlevel, aligninputchannel, minampl, maxampl,
              frequency, speedfactor, maxvelocity, maxdirectionchanges, speedoffset)
        checksize((1, 1, 1), routinename, scanaxis, stepaxis)
        cmdstr = self.getcmdstr('FDG', routinename, scanaxis, stepaxis)
        cmdstr += '' if minlevel is None else ' %s' % self.getcmdstr('ML', minlevel)
        cmdstr += '' if aligninputchannel is None else ' %s' % self.getcmdstr('A', aligninputchannel)
        cmdstr += '' if minampl is None else ' %s' % self.getcmdstr('MIA', minampl)
        cmdstr += '' if maxampl is None else ' %s' % self.getcmdstr('MAA', maxampl)
        cmdstr += '' if frequency is None else ' %s' % self.getcmdstr('F', frequency)
        cmdstr += '' if speedfactor is None else ' %s' % self.getcmdstr('SP', speedfactor)
        cmdstr += '' if maxvelocity is None else ' %s' % self.getcmdstr('V', maxvelocity)
        cmdstr += '' if maxdirectionchanges is None else ' %s' % self.getcmdstr('MDC', maxdirectionchanges)
        cmdstr += '' if speedoffset is None else ' %s' % self.getcmdstr('SPO', speedoffset)
        self.__msgs.send(cmdstr)

    # Too many local variables pylint: disable=R0914
    def FDR(self, routinename, scanaxis, scanrange, stepaxis, steprange, threshold=None, aligninputchannel=None,
            frequency=None, velocity=None, scanmiddlepos=None, stepmiddlepos=None, targettype=None,
            estimationmethod=None, mininput=None, maxinput=None, stopoption=None):
        """Define a fast alignment raster scan process.
        Use FRS to start the process and FRR? to read out the results of the process.
        Optional parameters need to be scalar, i.e. not list, tuple, dict.
        @param routinename : Name of the fast alignment routine to define, must be unique and != 0.
        @param scanaxis : Axis that is defined to be the master axis of the scan process.
        @param scanrange : Scan range for scan axis.
        @param stepaxis : Axis that is defined to be the second axis of the scan process.
        @param steprange : Scan range for step axis.
        @param threshold : If no value of the alignment signal to scan is higher than the given level,
        FRS? reports "not successful". Defaults to 0.1.
        @param aligninputchannel : Input channel to align, defaults to 1.
        @param frequency : Frequency of scan axis for sinusoidal scan , defaults to 10 Hz.
        @param velocity : Velocity of step axis that is driving a ramp, defaults to current velocity.
        @param scanmiddlepos : Middle position for scan axis. Defaults to the current position of the scan axis.
        @param stepmiddlepos : Middle position for step axis. Defaults to the current position of the step axis.
        @param targettype : Target signal type for raster scan, 0: sinusoidal type, 1:spiral scan type.
        @param estimationmethod : Data set will be post calculated, so that the theoretical maximum refers to a given
        distribution set, 0: optical maximum, 1: maximum referring to Gaussian distribution, 2: maximum referring
        to center of gravity calculation
        @param mininput : Values of recorded data set that should be taken above this percentage level.
        @param maxinput : Values of recorded data set that should be taken below this percentage level.
        @param stopoption : 0: move to scan and step axis position with the maximum value on the input channel,
        1: stay at endposition of scan process, 2: move to position at which process has been started, defaults to 0.
        """
        debug('GCSCommands.FDR(routinename=%r, scanaxis=%r, scanrange=%r, stepaxis=%r, steprange=%r, threshold=%r, '
              'aligninputchannel=%r, frequency=%r, velocity=%r, scanmiddlepos=%r, stepmiddlepos=%r, '
              'targettype=%r, estimationmethod=%r, mininput=%r, maxinput=%r, stopoption=%r)',
              routinename, scanaxis, scanrange, stepaxis, steprange, threshold, aligninputchannel, frequency, velocity,
              scanmiddlepos, stepmiddlepos, targettype, estimationmethod, mininput, maxinput, stopoption)
        checksize((1, 1, 1, 1, 1), routinename, scanaxis, scanrange, stepaxis, steprange)
        cmdstr = self.getcmdstr('FDR', routinename, scanaxis, scanrange, stepaxis, steprange)
        cmdstr += '' if threshold is None else ' %s' % self.getcmdstr('L', threshold)
        cmdstr += '' if aligninputchannel is None else ' %s' % self.getcmdstr('A', aligninputchannel)
        cmdstr += '' if frequency is None else ' %s' % self.getcmdstr('F', frequency)
        cmdstr += '' if velocity is None else ' %s' % self.getcmdstr('V', velocity)
        cmdstr += '' if scanmiddlepos is None else ' %s' % self.getcmdstr('MP1', scanmiddlepos)
        cmdstr += '' if stepmiddlepos is None else ' %s' % self.getcmdstr('MP2', stepmiddlepos)
        cmdstr += '' if targettype is None else ' %s' % self.getcmdstr('TT', targettype)
        cmdstr += '' if estimationmethod is None else ' %s' % self.getcmdstr('CM', estimationmethod)
        cmdstr += '' if mininput is None else ' %s' % self.getcmdstr('MIIL', mininput)
        cmdstr += '' if maxinput is None else ' %s' % self.getcmdstr('MAIL', maxinput)
        cmdstr += '' if stopoption is None else ' %s' % self.getcmdstr('ST', stopoption)
        self.__msgs.send(cmdstr)

    def FGC(self, process, scancenter, stepcenter):
        """Set fast gradient scan dither center.
        If only one axis is used for scan process, 'stepcenter' must be identical to 'scancenter'.
        @param process : Process ID as string or list.
        @param scancenter : Scan axis center value as float or list.
        @param stepcenter : Step axis center value as float or list.
        """
        debug('GCSCommands.FGC(process=%r, scancenter=%r, stepcenter=%r)', process, scancenter, stepcenter)
        checksize((True, True, True), process, scancenter, stepcenter)
        cmdstr = self.getcmdstr('FGC', process, scancenter, stepcenter)
        self.__msgs.send(cmdstr)

    def SIC(self, inputid, calctype, parameter=None):
        """Set fast alignment inputid calculation.
        @param inputid : Fast alignment input ID as integer.
        @param calctype : Calculation type as integer, 0: calculation off, 1: exponential calculation
        (parameter: a, b, c, d with a + b * c^(d*v)), 2: polynomial calculation (parameter: a0, a1, a2, a3, a4
        with a0 + a1*v^1 + a2*v^2 + a3*v^3 + a4*v^4, v is value after mechanical polynomial).
        @param parameter : See 'calctype'.
        """
        debug('GCSCommands.SIC(inputid=%r, calctype=%r, parameter=%r)', inputid, calctype, parameter)
        checksize((1, 1), inputid, calctype, parameter)
        cmdstr = self.getcmdstr('SIC', inputid, calctype, parameter)
        self.__msgs.send(cmdstr)

    def FRC(self, base, coupled):
        """Set fast alignment routine coupling.
        If coupling should be removed, 'coupled' has to be 0. No further arguments are allowed.
        @param base : Base routine ID as string.
        @param coupled : Coupled routine ID as string or list.
        """
        debug('GCSCommands.FRC(base=%r, coupled=%r)', base, coupled)
        checksize((1, True), base, coupled)
        cmdstr = self.getcmdstr('FRC', base, coupled)
        self.__msgs.send(cmdstr)

    def FRS(self, name):
        """Start the given fast alignment process (one or more).
        Process must be predefined, e.g. with FDR or FDG. The name of the process must be unique.
        @param name : Name of the routine as string or list.
        """
        debug('GCSCommands.FRS(name=%r)', name)
        checksize((True,), name)
        cmdstr = self.getcmdstr('FRS', name)
        self.__msgs.send(cmdstr)

    def FRP(self, name, option=None):
        """Stop, halt or resume the given fast alignment process(es).
        @param name : Name of the process as string or list or dict {name: option}.
        @param option : Option as integer or list, 0: stop the given process, 1: pause the given process,
        2: resume the given process.
        """
        debug('GCSCommands.FRP(name=%r, option=%r)', name, option)
        name, option = getitemsvaluestuple(name, option)
        cmdstr = self.getcmdstr('FRP', name, option)
        self.__msgs.send(cmdstr)

    def qFRP(self, name=None):
        """Return current status of given fast alignment process(es).
        @param name : Name of the process as string or list.
        @return : Ordered dictionary of {name: value}, values are integers, where 0: given process has been stopped or
        is not running, 1: given process has been paused, 2: given process is running, name is string.
        """
        debug('GCSCommands.qFRP(name=%r)', name)
        cmdstr = self.getcmdstr('FRP?', name)
        answer = self.__msgs.read(cmdstr)
        answerdict = getdict_oneitem(answer, name, valueconv=(int,), itemconv=str)
        debug('GCSCommands.qFRP = %r', answerdict)
        return answerdict

    def qFRC(self, base=None):
        """Return current fast alignment routine coupling.
        @param base : Base routine ID as string or list.
        @return : Ordered dictionary of {base: value}, values are strings of coupled routines.
        """
        debug('GCSCommands.qFRC(base=%r)', base)
        cmdstr = self.getcmdstr('FRC?', base)
        answer = self.__msgs.read(cmdstr)
        answerdict = getdict_oneitem(answer, base, valueconv=(str,))
        debug('GCSCommands.qFRC = %r', answerdict)
        return answerdict

    def qFGC(self, process=None):
        """Return fast gradient scan dither center.
        @param process : Process ID as string or list.
        @return : Ordered dictionary of {process: (scan, step)}, scan and step are floats.
        """
        debug('GCSCommands.qFGC(process=%r)', process)
        cmdstr = self.getcmdstr('FGC?', process)
        answer = self.__msgs.read(cmdstr)
        answerdict = getdict_oneitem(answer, process, valueconv=(float, float))
        debug('GCSCommands.qFGC = %r', answerdict)
        return answerdict

    def qFSF(self, axes=None):
        """Return parameters set with FSF().
        @param axes : String convertible or list of them or None.
        @return : Ordered dictionary of {axes: (force1, offset, force2)}, all values are floats.
        """
        debug('GCSCommands.qFSF(axes=%r)', axes)
        cmdstr = self.getcmdstr('FSF?', axes)
        answer = self.__msgs.read(cmdstr)
        answerdict = getdict_oneitem(answer, axes, valueconv=(float, float, float))
        debug('GCSCommands.qFSF = %r', answerdict)
        return answerdict

    def qSIC(self, inputid=None):
        """Return input signal calculation.
        @param inputid : Fast alignment input ID as integer or list.
        @return : Ordered dictionary of {inputid: (calctype, params)}, inputid, calctype are int, params are floats.
        """
        debug('GCSCommands.qSIC(inputid=%r)', inputid)
        cmdstr = self.getcmdstr('SIC?', inputid)
        answer = self.__msgs.read(cmdstr)
        answerdict = getdict_oneitem(answer, inputid, valueconv=(int, float), itemconv=int)
        debug('GCSCommands.qSIC = %r', answerdict)
        return answerdict

    def qFRR(self, name=None, resultid=None):
        """Return results of given fast alignment process.
        @param name : Name of fast alignment process as string or list of names or None for all.
        @param resultid : Identifier of the desired result as integer (see controller manual) or list of IDs or None.
        @return : Answer as dict {name: {resultid:value}}, name and value as string, resultid as int.
        """
        debug('GCSCommands.qFRR(name=%r, resultid=%r)', name, resultid)
        name, resultid = getitemsvaluestuple(name, resultid, required=False)
        if name:
            checksize((len(name),), resultid)
        cmdstr = self.getcmdstr('FRR?', name, resultid)
        answer = self.__msgs.read(cmdstr)
        answerdict = getdict_twoitems(answer, name, resultid, itemconv=[str, int], valueconv=(str,))
        debug('GCSCommands.qFRR = %r', answerdict)
        return answerdict

    def qTCI(self, inputid=None):
        """Return calculated fast alignment input value.
        @param inputid : Fast alignment input ID as integer or list.
        @return : Ordered dictionary of {inputid: value}, value is float.
        """
        debug('GCSCommands.qTCI(inputid=%r)', inputid)
        cmdstr = self.getcmdstr('TCI?', inputid)
        answer = self.__msgs.read(cmdstr)
        answerdict = getdict_oneitem(answer, inputid, valueconv=(float,), itemconv=int)
        debug('GCSCommands.qTCI = %r', answerdict)
        return answerdict

    def qFRH(self):
        """Return an help string with available query options for the FRR? command.
        @return : Help string.
        """
        debug('GCSCommands.qFRH()')
        answer = self.__msgs.read('FRH?')
        debug('GCSCommands.qFRH = %r', answer)
        return answer

    def qSAI(self):
        """Get the identifiers for all configured axes.
        Deactivated axes are not shown. Call qSAI_ALL() for all axes.
        @return : List of configured axes as string, whitespaces are removed.
        """
        debug('GCSCommands.qSAI()')
        answer = self.__msgs.read('SAI?')
        answer = splitparams(answer, self.isgcs2)
        debug('GCSCommands.qSAI = %r', answer)
        return answer

    def qHLP(self):
        """Get the help string from the controller.
        @return : Help message as string with trailing linefeed.
        """
        debug('GCSCommands.qHLP()')
        answer = self.__msgs.read('HLP?')
        debug('GCSCommands.qHLP = %r', answer)
        return answer

    def qPOS(self, axes=None):
        """Get the current positions of 'axes'.
        If no position sensor is present in your system, the response to qPOS() is not meaningful.
        To request the current position of input signal channels (sensors) in physical units,
        use qTSP() instead.
        @param axes : String convertible or list of them or None.
        @return : Ordered dictionary of {axis: value}, values are float.
        """
        debug('GCSCommands.qPOS(axes=%r)', axes)
        cmdstr = self.getcmdstr('POS?', axes)
        answer = self.__msgs.read(cmdstr)
        answerdict = getdict_oneitem(answer, axes, valueconv=(float,))
        debug('GCSCommands.qPOS = %r', answerdict)
        return answerdict

    def SVO(self, axes, values=None):
        """Set servo-control "on" or "off" (closed-loop/open-loop mode).
        @param axes: Axis or list of axes or dictionary {axis : value}.
        @param values : Bool or list of bools or None.
        """
        debug('GCSCommands.SVO(axes=%r, values=%r)', axes, values)
        axes, values = getitemsvaluestuple(axes, values)
        cmdstr = self.getcmdstr('SVO', axes, values)
        self.__msgs.send(cmdstr)

    def qSVO(self, axes=None):
        """Get the servo-control mode for 'axes'.
        @param axes : String convertible or list of them or None.
        @return : Ordered dictionary of {axis: value}, values are bool.
        """
        debug('GCSCommands.qSVO(axes=%r)', axes)
        cmdstr = self.getcmdstr('SVO?', axes)
        answer = self.__msgs.read(cmdstr)
        answerdict = getdict_oneitem(answer, axes, valueconv=(bool,))
        debug('GCSCommands.qSVO = %r', answerdict)
        return answerdict

    def qFSR(self, axes=None):
        """Get result of last surface detection.
        @param axes : String convertible or list of them or None.
        @return : Ordered dictionary of {axis: value}, values are bool.
        """
        debug('GCSCommands.qFSR(axes=%r)', axes)
        cmdstr = self.getcmdstr('FSR?', axes)
        answer = self.__msgs.read(cmdstr)
        answerdict = getdict_oneitem(answer, axes, valueconv=(bool,))
        debug('GCSCommands.qFSR = %r', answerdict)
        return answerdict

    def VAR(self, names, values=None):
        """Set variables 'names' to values'.
        The variable is present in RAM only.
        @param names: Item or list of names or dictionary {name : value} as string convertible.
        @param values : String convertible or list of them or None.
        """
        debug('GCSCommands.VAR(names=%r, values=%r)', names, values)
        names, values = getitemsvaluestuple(names, values)
        cmdstr = self.getcmdstr('VAR', names, values)
        self.__msgs.send(cmdstr)

    def VCO(self, axes, values=None):
        """Set velocity-control "on" or "off" for 'axes'.
        When velocity-control is "on", the corresponding axes will move with the currently valid
        velocity. That velocity can be set with VEL().
        @param axes: Axis or list of axes or dictionary {axis : value}.
        @param values : Bool or list of bools or None.
        """
        debug('GCSCommands.VCO(axes=%r, values=%r)', axes, values)
        axes, values = getitemsvaluestuple(axes, values)
        cmdstr = self.getcmdstr('VCO', axes, values)
        self.__msgs.send(cmdstr)

    def SPI(self, axes, values=None):
        """Set the pivot point coordinates for 'axes' in the volatile memory.
        @param axes: Axis or list of axes or dictionary {axis : value}.
        @param values : Float or list of floats or None.
        """
        debug('GCSCommands.SPI(axes=%r, values=%r)', axes, values)
        axes, values = getitemsvaluestuple(axes, values)
        cmdstr = self.getcmdstr('SPI', axes, values)
        self.__msgs.send(cmdstr)

    def SRA(self, axes, values=None):
        """Gear ratio setting for electronic gearing of 'axes'.
        The given ratio is applied when electronic gearing is enabled for the 'axes' which are
        then the slaves. The ratio is defined as Ratio = Travel of Master / Travel of Slave
        @param axes: Axis or list of axes or dictionary {axis : value}.
        @param values : Float or list of floats or None.
        """
        debug('GCSCommands.SRA(axes=%r, values=%r)', axes, values)
        axes, values = getitemsvaluestuple(axes, values)
        cmdstr = self.getcmdstr('SRA', axes, values)
        self.__msgs.send(cmdstr)

    def SSL(self, axes, values=None):
        """Set Soft Limit of 'axes' on or off.
        The values for the negative and positive soft limits must be set by NLM and PLM
        @param axes: Axis or list of axes or dictionary {axis : value}.
        @param values : Bool or list of bools or None.
        """
        debug('GCSCommands.SSL(axes=%r, values=%r)', axes, values)
        axes, values = getitemsvaluestuple(axes, values)
        cmdstr = self.getcmdstr('SSL', axes, values)
        self.__msgs.send(cmdstr)

    def RON(self, axes, values=None):
        """Set referencing mode for given 'axes'.
        Determines how to reference axes measured by incremental sensors.
        @param axes: Axis or list of axes or dictionary {axis : value}.
        @param values : Bool or list of bools or None.
        """
        debug('GCSCommands.RON(axes=%r, values=%r)', axes, values)
        axes, values = getitemsvaluestuple(axes, values)
        cmdstr = self.getcmdstr('RON', axes, values)
        self.__msgs.send(cmdstr)

    def KLD(self, csname, axes, values=None):
        """Define a levelling coordinate system (KLD-type).
        A coordinate system defined with this command is intended to
        eliminate hexapod misalignment which is known via an external
        measurable deviation.
        @param csname : Name of the coordinate system as string.
        @param axes: Axis or list of axes or dictionary {axis : value}.
        @param values : Float or list of floats or None.
        """
        debug('GCSCommands.KLD(csname=%r, axes=%r, values=%r)', csname, axes, values)
        axes, values = getitemsvaluestuple(axes, values)
        checksize((1, True, True), csname, axes, values)
        cmdstr = self.getcmdstr('KLD', csname, axes, values)
        self.__msgs.send(cmdstr)

    def KSB(self, csname, axes, values=None):
        """Define a base coordinate system (KSB-type).
        A coordinate system defined with this command is intended to preset a
        hexapod configuration. The KSB default "PI_BASE" coordinate system
        cannot be changed, but can be enabled/disabled by KEN. New KSB
        coordinate systems can be defined at any time and enabled with CCL 1.
        KSB can only rotate the coordinate system by 0, 90, 180, 270, -90,
        -180, -270 degrees.
        @param csname : Name of the coordinate system as string.
        @param axes: Axis or list of axes or dictionary {axis : value}.
        @param values : Float or list of floats or None.
        """
        debug('GCSCommands.KSB(csname=%r, axes=%r, values=%r)', csname, axes, values)
        axes, values = getitemsvaluestuple(axes, values)
        checksize((1, True, True), csname, axes, values)
        cmdstr = self.getcmdstr('KSB', csname, axes, values)
        self.__msgs.send(cmdstr)

    def KSD(self, csname, axes, values=None):
        """Define a new KSD-type coordinate system in order to set a
        "directed" swivel with the parameters X, Y, Z (relative to the
        hexapod platform). The coordinate system is rotated
        with the parameters U, V, W.
        @param csname : Name of the coordinate system as string.
        @param axes: Axis or list of axes or dictionary {axis : value}.
        @param values : Float or list of floats or None.
        """
        debug('GCSCommands.KSD(csname=%r, axes=%r, values=%r)', csname, axes, values)
        axes, values = getitemsvaluestuple(axes, values)
        checksize((1, True, True), csname, axes, values)
        cmdstr = self.getcmdstr('KSD', csname, axes, values)
        self.__msgs.send(cmdstr)

    def KST(self, csname, axes, values=None):
        """Define a new Tool coordinate system (KST-type).
        @param csname : Name of the coordinate system as string.
        @param axes: Axis or list of axes or dictionary {axis : value}.
        @param values : Float or list of floats or None.
        """
        debug('GCSCommands.KST(csname=%r, axes=%r, values=%r)', csname, axes, values)
        axes, values = getitemsvaluestuple(axes, values)
        checksize((1, True, True), csname, axes, values)
        cmdstr = self.getcmdstr('KST', csname, axes, values)
        self.__msgs.send(cmdstr)

    def KSW(self, csname, axes, values=None):
        """Define a new Work coordinate system (KSW-type).
        @param csname : Name of the coordinate system as string.
        @param axes: Axis or list of axes or dictionary {axis : value}.
        @param values : Float or list of floats or None.
        """
        debug('GCSCommands.KSW(csname=%r, axes=%r, values=%r)', csname, axes, values)
        axes, values = getitemsvaluestuple(axes, values)
        checksize((1, True, True), csname, axes, values)
        cmdstr = self.getcmdstr('KSW', csname, axes, values)
        self.__msgs.send(cmdstr)

    def MAT(self, variable, arg1, math, arg2):
        """Do some calculations and store result in variable.
        @param variable : Name of the variable the result will be stored as string convertible.
        @param arg1: First argument as number.
        @param math : Operator as string like +, -, *, /, MOD, AND, OR, XOR.
        @param arg2: Second argument as number.
        """
        debug('GCSCommands.MAT(variable=%r, arg1=%r, math=%r, arg2=%s)', variable, arg1, math, arg2)
        checksize((1, 1, 1, 1), variable, arg1, math, arg2)
        cmdstr = self.getcmdstr('MAT', variable, '=', arg1, math, arg2)
        self.__msgs.send(cmdstr)

    def FSF(self, axis, force1, offset, force2=None):
        """Do a surface detection.
        @param axis: Name of single axis as string.
        @param force1: Force value 1 as float.
        @param offset : Position offset after finding surface as float.
        @param force2: Force value 2 as float, optional.
        """
        debug('GCSCommands.FSF(axis=%r, force1=%r, offset=%r, force2=%s)', axis, force1, offset, force2)
        checksize((1, 1, 1), axis, force1, offset)
        if force2:
            checksize((1,), force2)
        cmdstr = self.getcmdstr('FSF', axis, force1, offset, force2)
        self.__msgs.send(cmdstr)

    def MAC_START(self, macro, args=''):
        """Start macro with name 'macro' with arguments.
        'args' stands for the value of a local variable contained in
        the macro. The sequence of the values in the input must correspond to
        the numbering of the appropriate local variables, starting with the
        value of the local variable 1. The individual values must be separated
        from each other with spaces. A maximum of 256 characters are permitted
        per function line. 'args' can be given directly or via the value of
        another variable. To find out what macros are available call qMAC().
        @param macro : Name of macro to start as string.
        @param args : Arguments to pass to the macro as string or list of strings.
        """
        debug('GCSCommands.MAC_START(macro=%r, args=%r)', macro, args)
        checksize((1,), macro)
        cmdstr = self.getcmdstr('MAC START', macro, args)
        self.__msgs.send(cmdstr)

    def MAC_BEG(self, item):
        """Put the DLL in macro recording mode.
        This function sets a flag in the library and effects the
        operation of other functions. Function will fail if already in recording
        mode. If successful, the commands that follow become part of the macro,
        so do not check error state unless FALSE is returned. End the recording
        with MAC_END.
        @param item : Item name as string.
        """
        debug('GCSCommands.MAC_BEG(item=%r)', item)
        checksize((1,), item)
        cmdstr = self.getcmdstr('MAC BEG', item)
        self.__msgs.send(cmdstr)

    def MAC_STOP(self, item):
        """Stop macro 'item'.
        @param item : Item name as string.
        """
        debug('GCSCommands.MAC_STOP(item=%r)', item)
        checksize((1,), item)
        cmdstr = self.getcmdstr('MAC STOP', item)
        self.__msgs.send(cmdstr)

    def MAC_DEL(self, item):
        """Delete macro with name 'item'.
        To find out what macros are available call qMAC().
        @param item : Item name as string.
        """
        debug('GCSCommands.MAC_DEL(item=%r)', item)
        checksize((1,), item)
        cmdstr = self.getcmdstr('MAC DEL', item)
        self.__msgs.send(cmdstr)

    def MEX(self, condition):
        """Send a macro end on 'condition' command (MEX).
        @param condition : Condition as string.
        """
        debug('GCSCommands.MEX(condition=%r)', condition)
        checksize((1,), condition)
        cmdstr = self.getcmdstr('MEX', condition)
        self.__msgs.send(cmdstr)

    def KSF(self, item):
        """Define a new KSF-type coordinate system based on the current pose
        (= position + orientation) of the Hexapod platform.
        @param item : Item name as string.
        """
        debug('GCSCommands.KSF(item=%r)', item)
        checksize((1,), item)
        cmdstr = self.getcmdstr('KSF', item)
        self.__msgs.send(cmdstr)

    def KEN(self, item):
        """Enable an already defined existing coordinate system, i.e. assign
        'enabled' state. KEN ZERO (or 0) will disable the current coordinate
        system. ZERO is the name of the root coordinate system. At command
        level 1 or higher it will disable also KLF/KLD coordinate systems.
        KEN ZERO will NOT disable the user defined KSB coordinate system.
        @param item : Item name as string.
        """
        debug('GCSCommands.KEN(item=%r)', item)
        checksize((1,), item)
        cmdstr = self.getcmdstr('KEN', item)
        self.__msgs.send(cmdstr)

    def KRM(self, item):
        """Remove a coordinate system. Removing a KLF/KLD/KSB coordinate
        system needs CCL 1.
        @param item : Item name as string.
        """
        debug('GCSCommands.KRM(item=%r)', item)
        checksize((1,), item)
        cmdstr = self.getcmdstr('KRM', item)
        self.__msgs.send(cmdstr)

    def KLF(self, item):
        """Define a levelling coordinate system (KLF-type). A coordinate
        system defined with this command is intended to eliminate hexapod
        misalignment by moving the hexapod manually to the aligned position.
        @param item : Item name as string.
        """
        debug('GCSCommands.KLF(item=%r)', item)
        checksize((1,), item)
        cmdstr = self.getcmdstr('KLF', item)
        self.__msgs.send(cmdstr)

    def INI(self, axes=None):
        """Initializes the motion control chip for 'axes'.
        The following actions are done by INI(): Writes the stage parameters which were loaded
        with CST() from the stage database to the controller. Switches the servo on. Sets
        reference mode to True, i.e. REF(), FRF(), MNL(), FNL(), MPL() or FPL() is required to
        reference the axis, usage of POS() is not allowed. Sets reference state to "not
        referenced". If the stage has tripped a limit switch, INI() will move it away from the
        limit switch until the limit condition is no longer given, and the target position is set
        to the current position afterwards. Sets trigger output mode to default configuration.
        @param axes : String convertible or list of them or None.
        """
        debug('GCSCommands.INI(axes=%r)', axes)
        cmdstr = self.getcmdstr('INI', axes)
        self.__msgs.send(cmdstr)

    def IsMoving(self, axes=None):
        """Check if 'axes' are moving.
        If an axis is moving the corresponding element will be True, otherwise False.
        @param axes : String convertible or list of them or None.
        @return : Ordered dictionary of {axis: value}, values are bool.
        """
        debug('GCSCommands.IsMoving(axes=%r)', axes)
        checksize((), axes)
        answer = self.__msgs.read(chr(5))
        value = int(answer.strip(), base=16)
        answerdict = getbitcodeditems(value, self.allaxes, axes)
        debug('GCSCommands.IsMoving = %r', answerdict)
        return answerdict

    def IsGeneratorRunning(self, wavegens=None):
        """Get the status of the wave generator(s), the user profile mode or scan algorithm.
        @param wavegens : Integer convertible or list of them or None.
        @return : Ordered dictionary of {wavegen: value}, wavegens are int, values are bool.
        """
        debug('GCSCommands.IsGeneratorRunning(wavegens=%r)', wavegens)
        checksize((), wavegens)
        answer = self.__msgs.read(chr(9))
        value = int(answer.strip(), base=16)
        answerdict = getbitcodeditems(value, items=wavegens)
        debug('GCSCommands.IsGeneratorRunning = %r', answerdict)
        return answerdict

    def GetDynamicMoveBufferSize(self):
        """Get the free memory space of a buffer that contains the motion profile points.
        Corresponds to GCS command "#11".
        @return : Value as integer.
        """
        debug('GCSCommands.GetDynamicMoveBufferSize()')
        answer = self.__msgs.read(chr(11))
        value = int(answer.strip())
        debug('GCSCommands.GetDynamicMoveBufferSize = %r', value)
        return value

    def KCP(self, cssource, csdest=None):
        """Copy a coordinate system. Use in cases where coordinate systems should be linked but
        also be accessable in its original form.
        @param cssource: Name of source CS or list of them or dictionary {source : dest} as string.
        @param csdest : Name of destination CS as string convertible or list of them or None.
        """
        debug('GCSCommands.KCP(cssource=%r, csdest=%r)', cssource, csdest)
        cssource, csdest = getitemsvaluestuple(cssource, csdest)
        cmdstr = self.getcmdstr('KCP', cssource, csdest)
        self.__msgs.send(cmdstr)

    def KLN(self, childs, parents=None):
        """Link two coordinate systems together by defining a parent-child relation forming a
        chain.
        @param childs: Name of child CS or list of them or dictionary {source : dest} as string.
        @param parents : Name of parent CS or list of them as string or None.
        """
        debug('GCSCommands.KLN(childs=%r, parents=%r)', childs, parents)
        childs, parents = getitemsvaluestuple(childs, parents)
        cmdstr = self.getcmdstr('KLN', childs, parents)
        self.__msgs.send(cmdstr)

    def HLT(self, axes=None, noraise=False):
        """Halt the motion of given 'axes' smoothly.
        Error code 10 is set. HLT() does stop any motion that is caused by motion commands.
        @param axes : String convertible or list of them or None.
        @param noraise : If True a GCS error 10 (controller was stopped by command) will not be raised.
        """
        debug('GCSCommands.HLT(axes=%r)', axes)
        cmdstr = self.getcmdstr('HLT', axes)
        try:
            self.__msgs.send(cmdstr)
        except pipython.GCSError as exc:
            if noraise and exc == pipython.gcserror.E10_PI_CNTR_STOP:
                debug('GCS error 10 is masked')
            else:
                raise

    def IFC(self, items, values=None):
        """Set the interface configuration for parameter 'item'.
        After IFC() is sent, the new setting becomes active and the
        host PC interface configuration may need to be changed to maintain
        communication.
        @param items: Item or list of items or dictionary {item : value} as string convertible.
        @param values : String convertible or list of them or None.
        """
        debug('GCSCommands.IFC(items=%r, values=%r)', items, values)
        items, values = getitemsvaluestuple(items, values)
        cmdstr = self.getcmdstr('IFC', items, values)
        self.__msgs.send(cmdstr)

    def WGC(self, wavegens, numcycles=None):
        """Set the number of cycles for wave generator output.
        @param wavegens: Wavegen or list of them or dict {wavegen : numcycle} as int convertible.
        @param numcycles : Integer convertible or list of them or None.
        """
        debug('GCSCommands.WGC(wavegens=%r, numcycles=%r)', wavegens, numcycles)
        wavegens, numcycles = getitemsvaluestuple(wavegens, numcycles)
        cmdstr = self.getcmdstr('WGC', wavegens, numcycles)
        self.__msgs.send(cmdstr)

    def WGO(self, wavegens, mode=None):
        """Start and stop the specified wave generator with the given mode.
        @param wavegens: Wavegen or list of them or dict {wavegen : mode} as int convertible.
        @param mode : Integer convertible or list of them or None.
        """
        debug('GCSCommands.WGO(wavegens=%r, mode=%r)', wavegens, mode)
        wavegens, mode = getitemsvaluestuple(wavegens, mode)
        cmdstr = self.getcmdstr('WGO', wavegens, mode)
        self.__msgs.send(cmdstr)

    def WMS(self, tables, lengths=None):
        """Sets the maximum 'lengths' of the wave storage for 'tables'.
        @param tables: Wavetable or list of them or dict {table : length} as int convertible.
        @param lengths : Integer convertible or list of them or None.
        """
        debug('GCSCommands.WMS(tables=%r, lengths=%r)', tables, lengths)
        tables, lengths = getitemsvaluestuple(tables, lengths)
        cmdstr = self.getcmdstr('WMS', tables, lengths)
        self.__msgs.send(cmdstr)

    def IFS(self, password, items, values=None):
        """Interface parameter store in EPROM.
        The power-on default parameters for the interface are changed in non-volatile memory, but
        the current active parameters are not. Settings made become active with the next power-on
        or reboot.
        @param password : String convertible, usually "100".
        @param items: Item or list of items or dictionary {item : value}.
        @param values : String convertible or list of them or None.
        """
        debug('GCSCommands.IFS(password=%r, items=%r, values=%r)', password, items, values)
        items, values = getitemsvaluestuple(items, values)
        checksize((1, True, True), password, items, values)
        cmdstr = self.getcmdstr('IFS', password, items, values)
        self.__msgs.send(cmdstr)

    def WPA(self, password='100', items=None, params=None, checkerror=None):
        """Write the current settings from the volatile to the nonvolatile memory.
        @param password : String convertible, defaults to "100".
        @param items: Item or list of items or dictionary {item : param} or None.
        @param params : Parameter IDs as integer convertibles or list of them or None.
        @param checkerror : Defaults to None, if True or False the errcheck property is changed accordingly.
        """
        debug('GCSCommands.WPA(password=%r, items=%r, params=%r, checkerror=%r)', password, items, params, checkerror)
        items, params = getitemsvaluestuple(items, params, required=False)
        checksize((1,), password, items, params)
        errbuf = self.__msgs.errcheck
        if checkerror is not None and checkerror != errbuf:
            self.__msgs.errcheck = bool(checkerror)
        cmdstr = self.getcmdstr('WPA', password, items, params)
        self.__msgs.send(cmdstr)
        if checkerror is not None and checkerror != errbuf:
            self.__msgs.errcheck = errbuf

    def DPA(self, password='100', items=None, params=None):
        """Reset volatile parameters to default values.
        @param password : String convertible, defaults to "100".
        @param items: String convertible or list of them or dictionary {item : param} or None.
        @param params : Integer convertible or list of them or None.
        """
        debug('GCSCommands.DPA(password=%r, items=%r, params=%r)', password, items, params)
        items, params = getitemsvaluestuple(items, params, required=False)
        checksize((1,), password, items, params)
        cmdstr = self.getcmdstr('DPA', password, items, params)
        self.__msgs.send(cmdstr)

    def HasPosChanged(self, axes=None):
        """Corresponds to "#6".
        Queries whether the 'axes' positions have changed since the last position query was sent.
        @param axes : String convertible or list of them or None.
        @return : Ordered dictionary of {axis: value}, values are bool.
        """
        debug('GCSCommands.HasPosChanged(axes=%r)', axes)
        checksize((), axes)
        answer = self.__msgs.read(chr(6))
        value = int(answer.strip(), base=16)
        answerdict = getbitcodeditems(value, self.allaxes, axes)
        debug('GCSCommands.HasPosChanged = %r', answerdict)
        return answerdict

    def HIN(self, axes, values=None):
        """Activate HID control for controller axis.
        The HID device is connected to the controllers USB port.
        @param axes: Axis or list of axes or dictionary {axis : value}.
        @param values : Bool or list of bools or None.
        """
        debug('GCSCommands.HIN(axes=%r, values=%r)', axes, values)
        axes, values = getitemsvaluestuple(axes, values)
        cmdstr = self.getcmdstr('HIN', axes, values)
        self.__msgs.send(cmdstr)

    def GOH(self, axes=None):
        """Move all 'axes' to their home positions.
        This is equivalent to moving the axes to positions 0 using
        MOV(). Depending on the controller, the definition of the home position
        can be changed with DFH().
        @param axes : String convertible or list of them or None.
        """
        debug('GCSCommands.GOH(axes=%r)', axes)
        cmdstr = self.getcmdstr('GOH', axes)
        self.__msgs.send(cmdstr)

    def FED(self, axes, edgetype, param=None):
        """Move 'axes' to signal 'edgetype'.
        The following edge types are available:
        - 1 = negative limit switch
        - 2 = positive limit switch
        - 3 = reference switch
        @param axes: Axis or list of axes.
        @param edgetype : Integer convertible or list of them.
        @param param : Optional list of int, but usually None because it is not needed.
        """
        debug('GCSCommands.FED(axes=%r, edgetype=%r, param=%r)', axes, edgetype, param)
        axes = getitemslist(axes)
        if param is None:
            param = [0] * len(axes)
        checksize((True, True, True), axes, edgetype, param)
        cmdstr = self.getcmdstr('FED', axes, edgetype, param)
        self.__msgs.send(cmdstr)

    def FNL(self, axes=None):
        """Start a reference move to the negative limit switch.
        Moves all 'axes' synchronously to the negative physical limits
        of their travel ranges and sets the current positions to the negative
        range limit values. Call IsControllerReady() to find out if referencing
        is complete (the controller will be "busy" while referencing, so most
        other commands will cause a PI_CONTROLLER_BUSY error) and qFRF() to
        check whether the reference move was successful.
        Error check will be disabled temporarily for GCS1 devices.
        @param axes : String convertible or list of them or None.
        """
        debug('GCSCommands.FNL(axes=%r)', axes)
        errcheck = self.__msgs.errcheck
        if not self.isgcs2:
            self.__msgs.errcheck = False
        cmdstr = self.getcmdstr('FNL', axes)
        self.__msgs.send(cmdstr)
        if not self.isgcs2:
            self.__msgs.errcheck = errcheck

    def FPH(self, axes):
        """Find Phase.
        Find offset between motor and encoder by performing a homing process.
        Attention: The stage will start moving. Servo must be disabled (openloop).
        @param axes : String convertible or list of them.
        """
        debug('GCSCommands.FPH(axes=%r)', axes)
        checksize((True,), axes)
        cmdstr = self.getcmdstr('FPH', axes)
        self.__msgs.send(cmdstr)

    def FPL(self, axes=None):
        """Start a reference move to the positive limit switch.
        Moves all 'axes' synchronously to the positive physical limits
        of their travel ranges and sets the current positions to the positive
        range limit values. Call IsControllerReady() to find out if referencing
        is complete (the controller will be "busy" while referencing, so most
        other commands will cause a PI_CONTROLLER_BUSY error) and qFRF() to
        check whether the reference move was successful.
        Error check will be disabled temporarily for GCS1 devices.
        @param axes : String convertible or list of them or None.
        """
        debug('GCSCommands.FPL(axes=%r)', axes)
        errcheck = self.__msgs.errcheck
        if not self.isgcs2:
            self.__msgs.errcheck = False
        cmdstr = self.getcmdstr('FPL', axes)
        self.__msgs.send(cmdstr)
        if not self.isgcs2:
            self.__msgs.errcheck = errcheck

    def FRF(self, axes=None):
        """Start a reference move to the reference switch.
        Moves all 'axes' synchronously to the physical reference point
        and sets the current positions to the reference position. Call
        IsControllerReady() to find out if referencing is complete (the
        controller will be "busy" while referencing, so most other commands
        will cause a PI_CONTROLLER_BUSY error) and qFRF() to check whether the
        reference move was successful.
        Error check will be disabled temporarily for GCS1 devices.
        @param axes : String convertible or list of them or None.
        """
        debug('GCSCommands.FRF(axes=%r)', axes)
        errcheck = self.__msgs.errcheck
        if not self.isgcs2:
            self.__msgs.errcheck = False
        cmdstr = self.getcmdstr('FRF', axes)
        self.__msgs.send(cmdstr)
        if not self.isgcs2:
            self.__msgs.errcheck = errcheck

    def DPO(self, axes=None):
        """Dynamic Digital Linearization (DDL) Parameter Optimization.
        Recalculates the internal DDL processing parameters
        (Time Delay Max, ID 0x14000006, Time Delay Min, ID 0x14000007) for specified 'axes'.
        @param axes : String convertible or list of them or None.
        """
        debug('GCSCommands.DPO(axes=%r)', axes)
        cmdstr = self.getcmdstr('DPO', axes)
        self.__msgs.send(cmdstr)

    def TRI(self, lines, values=None):
        """Enable or disable the trigger input mode which was set with CTO().
        @param lines: Lines as int or list of them or dictionary {line : value}.
        @param values : Bool convertible or list of them or None.
        """
        debug('GCSCommands.TRI(lines=%r, values=%r)', lines, values)
        lines, values = getitemsvaluestuple(lines, values)
        cmdstr = self.getcmdstr('TRI', lines, values)
        self.__msgs.send(cmdstr)

    def TRO(self, lines, values=None):
        """Enable or disable the trigger output mode which was set with CTO().
        @param lines: Lines as int or list of them or dictionary {line : value}.
        @param values : Bool convertible or list of them or None.
        """
        debug('GCSCommands.TRO(lines=%r, values=%r)', lines, values)
        lines, values = getitemsvaluestuple(lines, values)
        cmdstr = self.getcmdstr('TRO', lines, values)
        self.__msgs.send(cmdstr)

    def TSP(self, channels, values=None):
        """Set current sensor position. Only possible when the reference mode is switched off, see RON().
        @param channels: Sensor channel number as int or list of them or dictionary {channel : value}.
        @param values : Float convertible or list of them or None.
        """
        debug('GCSCommands.TSP(channels=%r, values=%r)', channels, values)
        channels, values = getitemsvaluestuple(channels, values)
        cmdstr = self.getcmdstr('TSP', channels, values)
        self.__msgs.send(cmdstr)

    def WSL(self, wavegens, tables=None):
        """Wave table selection.
        Connects a wave table in 'tables' to a wave generator in
        'wavegens' or disconnects the selected generator from any wave table.
        Two or more generators can be connected to the same wave table, but a
        generator cannot be connected to more than one wave table. Deleting
        wave table content with WCL() has no effect on the WSL() settings. As
        long as a wave generator is running, it is not possible to change the
        connected wave table.
        @param wavegens: Lines as int or list of them or dictionary {wavegen : value}.
        @param tables : Wave table IDs as integer or list of them or None.
        Zero disconnects the wave generator from any wave tables.
        """
        debug('GCSCommands.WSL(wavegens=%r, tables=%r)', wavegens, tables)
        wavegens, tables = getitemsvaluestuple(wavegens, tables)
        cmdstr = self.getcmdstr('WSL', wavegens, tables)
        self.__msgs.send(cmdstr)

    def DFH(self, axes=None):
        """Define the current positions of 'axes' as the axis home position
        by setting the position value to 0.00.
        @param axes : String convertible or list of them or None.
        """
        debug('GCSCommands.DFH(axes=%r)', axes)
        cmdstr = self.getcmdstr('DFH', axes)
        self.__msgs.send(cmdstr)

    def DCO(self, axes, values=None):
        """Set drift compensation mode for 'axes' (on or off).
        Drift compensation is applied to avoid unwanted changes in displacement over time and is
        therefore recommended for static operation.
        @param axes: Axis or list of axes or dictionary {axis : value}.
        @param values : Bool or list of bools or None.
        """
        debug('GCSCommands.DCO(axes=%r, values=%r)', axes, values)
        axes, values = getitemsvaluestuple(axes, values)
        cmdstr = self.getcmdstr('DCO', axes, values)
        self.__msgs.send(cmdstr)

    def EAX(self, axes, values=None):
        """Enable axis.
        Affected: motion commands like MOV, MVR, SMO, STE, IMP, SVA, OMA, OMR, MRT, MRW, and
        wavegen, macros, analog input, joystick. "EAX 0" does not imply that the motor current is
        zero. During motion "EAX 0" will do the same as "STP". A motion error will disable EAX, ECH
        and SVO. "SVO" will not disable/enable "EAX". "EAX 1" will not enable SVO. But "EAX 0" will
        disable "SVO". SVO can be enabled only if EAX is enabled. There is a parameter
        "Enable EAX on startup".
        @param axes: Axis or list of axes or dictionary {axis : value}.
        @param values : Bool or list of bools or None.
        """
        debug('GCSCommands.EAX(axes=%r, values=%r)', axes, values)
        axes, values = getitemsvaluestuple(axes, values)
        cmdstr = self.getcmdstr('EAX', axes, values)
        self.__msgs.send(cmdstr)

    def ATZ(self, axes=None, lowvoltage=None):
        """Start an appropriate calibration procedure for 'axes'. The AutoZero procedure will move
        the axis, and the motion may cover the whole travel range. Make sure that it is safe for
        the stage to move.
        @param axes: Axis or list of axes or dictionary {axis : lowvoltage}.
        @param lowvoltage : Float or list of floats or None.
        """
        debug('GCSCommands.ATZ(axes=%r, lowvoltage=%r)', axes, lowvoltage)
        axes, lowvoltage = getitemsvaluestuple(axes, lowvoltage, required=False)
        cmdstr = self.getcmdstr('ATZ', axes, lowvoltage)
        self.__msgs.send(cmdstr)

    def CTI(self, lines, params, values):
        """Configure the trigger conditions for the given digital input 'lines'.
        The defined trigger input actions only become active when enabled with TRI.
        @param lines : Digital input lines as list of integer convertible.
        @param params : Parameter IDs as list of integer convertible.
        @param values : Parameter values as list of integer convertible.
        """
        debug('GCSCommands.CTI(lines=%r, params=%r, values=%r)', lines, params, values)
        checksize((True, True, True), lines, params, values)
        cmdstr = self.getcmdstr('CTI', lines, params, values)
        self.__msgs.send(cmdstr)

    def DDL(self, table, offsets, values):
        """Load the Dynamic Digital Linearization data to the learning 'tables'.
        @param table : Digital input table as integer convertible.
        @param offsets : Parameter ID as integer convertible.
        @param values : Parameter values as list of integer convertible.
        """
        debug('GCSCommands.DDL(table=%r, offsets=%r, values=%r)', table, offsets, values)
        checksize((1, 1, True), table, offsets, values)
        cmdstr = self.getcmdstr('DDL', table, offsets, values)
        self.__msgs.send(cmdstr)

    def DRT(self, tables, sources, values=None):
        """Define a trigger source for data recorder 'tables'.
        If table == 0 the specified trigger source is set for all data recorder tables.
        @param tables : Data recorder table as integer convertible or list of them.
        @param sources : Trigger source IDs as integer convertible or list of them.
        @param values : Trigger source values as string or list or None for default.
        """
        debug('GCSCommands.DRT(tables=%r, sources=%r, values=%r)', tables, sources, values)
        tables = getitemslist(tables)
        if values is None:
            values = values or [0] * len(tables)
        checksize((True, True, True), tables, sources, values)
        cmdstr = self.getcmdstr('DRT', tables, sources, values)
        self.__msgs.send(cmdstr)

    def WTR(self, wavegens, tablerates, interpol):
        """Set wave generator table rate.
        @param wavegens : Generator ID as integer convertible or list of them.
        @param tablerates : Duration of a wave table point in multiples of servo cycles
        as integer convertible or list of them.
        @param interpol : Interpolation type as integer convertible or list of them.
        """
        debug('GCSCommands.WTR(wavegens=%r, tablerates=%r, interpol=%r)', wavegens, tablerates,
              interpol)
        checksize((True, True, True), wavegens, tablerates, interpol)
        cmdstr = self.getcmdstr('WTR', wavegens, tablerates, interpol)
        self.__msgs.send(cmdstr)

    def TWS(self, lines, points, switches):
        """Set trigger line actions to waveform points for the given trigger output line.
        For the selected trigger output line the generator trigger mode must be activated by
        CTO(). Does not clear existing definition.
        @param lines : Trigger line ID as integer convertible or list of them.
        @param points : Wave table point index as integer convertible or list of them.
        @param switches : Value of wave table point as integer convertible or list of them.
        """
        debug('GCSCommands.TWS(lines=%r, points=%r, switches=%r)', lines, points, switches)
        checksize((True, True, True), lines, points, switches)
        cmdstr = self.getcmdstr('TWS', lines, points, switches)
        self.__msgs.send(cmdstr)

    def MAC_NSTART(self, macro, numruns, params=None):
        """Repeat 'macro' 'numruns' times and optionally pass 'params' as macro arguments.
        Another execution is started when the last one is finished.
        @param macro : Name of the macro as string.
        @param numruns : Number of runs as integer.
        @param params : Macro arguments as string convertible or list of them.
        """
        debug('GCSCommands.MAC_NSTART(macro=%r, numruns=%r, params=%r)', macro, numruns, params)
        checksize((1, 1), macro, numruns, params)
        cmdstr = self.getcmdstr('MAC NSTART', macro, numruns, params)
        self.__msgs.send(cmdstr)

    def MAC_qDEF(self):
        """Get name of default or startup macro.
        @return : Macro name with trailing linefeed.
        """
        debug('GCSCommands.MAC_qDEF()')
        answer = self.__msgs.read('MAC DEF?')
        debug('GCSCommands.MAC_qDEF = %r', answer)
        return answer

    def IsRunningMacro(self):
        """Test if a macro is running, corresponds to GCS command "#8".
        @return : True if a macro is running.
        """
        debug('GCSCommands.IsRunningMacro()')
        answer = self.__msgs.read(chr(8))
        answer = convertvalue(answer, bool)
        debug('GCSCommands.IsRunningMacro = %r', answer)
        return answer

    def IsControllerReady(self):
        """Test if controller is ready, corresponds to GCS command "#7". No error check.
        @return : True if controller is ready.
        """
        debug('GCSCommands.IsControllerReady()')
        errcheck = self.__msgs.errcheck
        self.__msgs.errcheck = False
        answer = self.__msgs.read(chr(7))
        self.__msgs.errcheck = errcheck
        try:
            if ord(answer.strip()) == 177:
                answer = True
            elif ord(answer.strip()) == 176:
                answer = False
            else:
                raise TypeError
        except TypeError:
            raise ValueError('unexpected response %r for IsControllerReady()' % answer)
        debug('GCSCommands.IsControllerReady = %r', answer)
        return answer

    def MAC_qERR(self):
        """Get error occured during macro execution.
        @return : Error as string with trailing linefeed.
        """
        debug('GCSCommands.MAC_qERR()')
        answer = self.__msgs.read('MAC ERR?')
        debug('GCSCommands.MAC_qERR = %r', answer)
        return answer

    def HDT(self, devices, axes, values):
        """Set human interface device (HID) default lookup table.
        @param devices : HID device ID as integer convertible or list of them.
        @param axes : HID axis ID as integer convertible or list of them.
        @param values : Lookup table ID (see manual) as integer convertible or list of them.
        """
        debug('GCSCommands.HDT(devices=%r, axes=%r, values=%r)', devices, axes, values)
        checksize((True, True, True), devices, axes, values)
        cmdstr = self.getcmdstr('HDT', devices, axes, values)
        self.__msgs.send(cmdstr)

    def TWE(self, tables, start, end):
        """Define the edges of a trigger signal which is to be output in conjunction with the wave generator output.
        @param tables : Table ID as integer convertible or list of them.
        @param start : Wave table index where trigger starts as integer convertible or list of them.
        @param end : Wave table index where trigger ends as integer convertible or list of them.
        """
        debug('GCSCommands.TWE(tables=%r, start=%r, end=%r)', tables, start, end)
        checksize((True, True, True), tables, start, end)
        cmdstr = self.getcmdstr('TWE', tables, start, end)
        self.__msgs.send(cmdstr)

    def HIL(self, devices, leds, values):
        """Set state of human interface device (HID) LEDs.
        @param devices : HID device ID as integer convertible or list of them.
        @param leds : HID LED ID as integer convertible or list of them.
        @param values : LED mode ID (see manual) as integer convertible or list of them.
        """
        debug('GCSCommands.HIL(devices=%r, leds=%r, values=%r)', devices, leds, values)
        checksize((True, True, True), devices, leds, values)
        cmdstr = self.getcmdstr('HIL', devices, leds, values)
        self.__msgs.send(cmdstr)

    def HIS(self, devices, items, properties, values):
        """Set value of item of human interface device (HID).
        @param devices : HID device ID as integer convertible or list of them.
        @param items : HID item ID as integer convertible or list of them.
        @param properties : HID property ID as integer convertible or list of them.
        @param values : Property value to set as integer convertible or list of them.
        """
        debug('GCSCommands.HIS(devices=%r, items=%r, properties=%r, values=%r)', devices, items,
              properties, values)
        checksize((True, True, True, True), devices, items, properties, values)
        cmdstr = self.getcmdstr('HIS', devices, items, properties, values)
        self.__msgs.send(cmdstr)

    def HIT(self, tables, index, values):
        """Set value of customer lookup table for a human interface device (HID).
        @param tables : HID lookup table ID as integer convertible or list of them.
        @param index : Index of point in table as integer convertible or list of them.
        @param values : Value of point to set as float convertible or list of them.
        """
        debug('GCSCommands.HIT(tables=%r, index=%r, values=%r)', tables, index, values)
        checksize((True, True, True), tables, index, values)
        cmdstr = self.getcmdstr('HIT', tables, index, values)
        self.__msgs.send(cmdstr)

    def JDT(self, devices, devaxes, values):
        """Set joystick default lookup table.
        @param devices : Joystick device ID as integer convertible or list of them.
        @param devaxes : Joystick axis ID as integer convertible or list of them.
        @param values : Type of lookup table (see manual) as integer convertible or list of them.
        """
        debug('GCSCommands.JDT(devices=%r, devaxes=%r, values=%r)', devices, devaxes, values)
        checksize((True, True, True), devices, devaxes, values)
        cmdstr = self.getcmdstr('JDT', devices, devaxes, values)
        self.__msgs.send(cmdstr)

    def JAX(self, device, devaxis, axes=''):
        """Set joystick controlled 'axes'.
        @param device : Single joystick device ID as integer convertible.
        @param devaxis : Single joystick axis ID as integer convertible.
        @param axes : Controller axes as string convertible or list of them or None.
        """
        debug('GCSCommands.JAX(device=%r, devaxis=%r, axes=%r)', device, devaxis, axes)
        checksize((1, 1), device, devaxis)
        cmdstr = self.getcmdstr('JAX', device, devaxis, axes)
        self.__msgs.send(cmdstr)

    def JON(self, devices, values=None):
        """Enable joystick control.
        @param devices : Joystick device ID as integer or list of them or dict {device: value}.
        @param values : True to enable, bool convertible or list of them.
        """
        debug('GCSCommands.JON(devices=%r, values=%r)', devices, values)
        devices, values = getitemsvaluestuple(devices, values)
        cmdstr = self.getcmdstr('JON', devices, values)
        self.__msgs.send(cmdstr)

    def HIA(self, axes, functions, devices, devaxes):
        """Assign human interface device (HID) axis to a function of a controller axis.
        @param axes : Controller axis name as string convertible or list of them.
        @param functions : Function ID (see manual) as integer convertible or list of them.
        @param devices : HID device ID as integer convertible or list of them.
        @param devaxes : HID axis ID as integer convertible or list of them.
        """
        debug('GCSCommands.HIA(axes=%r, functions=%r, devices=%r, devaxes=%r)', axes, functions, devices, devaxes)
        checksize((True, True, True, True), axes, functions, devices, devaxes)
        cmdstr = self.getcmdstr('HIA', axes, functions, devices, devaxes)
        self.__msgs.send(cmdstr)

    def WAV_NOISE(self, table, append, amplitude, offset, seglength):
        """Define waveform of type "noise".
        @param table : Wave table ID as integer convertible.
        @param append : "X" to start from first point, "&" to append and "+" to add to existing waveform.
        @param amplitude : Amplitude of the noise as float.
        @param offset : Offset of the noise as float.
        @param seglength : Length of the segment as integer.
        """
        debug('GCSCommands.WAV_NOISE(table=%r, append=%r, amplitude=%r, offset=%r, seglength=%r)', table,
              append, amplitude, offset, seglength)
        checksize((1, 1, 1, 1, 1), table, append, amplitude, offset, seglength)
        cmdstr = self.getcmdstr('WAV', table, append, 'NOISE', seglength, amplitude, offset)
        self.__msgs.send(cmdstr)

    def MOD(self, items, modes, values):
        """Set modes.
        @param items : Axes or channels as string convertible or list of them.
        @param modes : Modes to modify as integer convertible or list of them.
        @param values : Values for modes to set as string convertible or list of them.
        """
        debug('GCSCommands.MOD(items=%r, modes=%r, values=%r)', items, modes, values)
        checksize((True, True, True), items, modes, values)
        cmdstr = self.getcmdstr('MOD', items, modes, values)
        self.__msgs.send(cmdstr)

    def SWT(self, item, index, value):
        """Set wave table data.
        @param item : Controller index as string convertible.
        @param index : Index of wave table point to set as integer.
        @param value : Value to set as float.
        """
        debug('GCSCommands.SWT(item=%r, index=%r, value=%r)', item, index, value)
        checksize((1, 1, 1), item, index, value)
        cmdstr = self.getcmdstr('SWT', item, index, value)
        self.__msgs.send(cmdstr)

    def WTO(self, item, value, timer):
        """Enable wave table output.
        @param item : Axis or channel as string convertible.
        @param value : Number of wave data to use as integer.
        @param timer : Timer value in milliseconds as integer or list of them.
        """
        debug('GCSCommands.WTO(item=%r, value=%r, timer=%r)', item, value, timer)
        checksize((1, 1, True), item, value, timer)
        cmdstr = self.getcmdstr('WTO', item, value, timer)
        self.__msgs.send(cmdstr)

    def MNL(self, axes=None):
        """Move 'axes' to negative limit switch.
        @param axes : Axes as string convertible or list of them.
        """
        debug('GCSCommands.MNL(axes=%r)', axes)
        cmdstr = self.getcmdstr('MNL', axes)
        self.__msgs.send(cmdstr)

    def MPL(self, axes=None):
        """Move 'axes' to positive limit switch.
        @param axes : Axes as string convertible or list of them.
        """
        debug('GCSCommands.MPL(axes=%r)', axes)
        cmdstr = self.getcmdstr('MPL', axes)
        self.__msgs.send(cmdstr)

    def RST(self, axes=None):
        """Restore stage configuration and motion parameters which were last saved with SAV().
        In contrast to ITD(), RST() changes the stage-to-axis assignment.
        @param axes : Axes as string convertible or list of them.
        """
        debug('GCSCommands.RST(axes=%r)', axes)
        cmdstr = self.getcmdstr('RST', axes)
        self.__msgs.send(cmdstr)

    def ITD(self, axes=None):
        """Initialize stage configuration and motion parameters to default values.
        In contrast to RST(), ITD() does not change the stage-to-axis assignment.
        @param axes : Axes as string convertible or list of them.
        """
        debug('GCSCommands.ITD(axes=%r)', axes)
        cmdstr = self.getcmdstr('ITD', axes)
        self.__msgs.send(cmdstr)

    def RTO(self, axes=None):
        """Set device ready to turn off.
        The current position of given axis is written to non-volatile memory. When the controller
        is switched on next time, the saved position is read from non-volatile memory and set as
        current position.
        @param axes : Axes as string convertible or list of them.
        """
        debug('GCSCommands.RTO(axes=%r)', axes)
        cmdstr = self.getcmdstr('RTO', axes)
        self.__msgs.send(cmdstr)

    def SCH(self, axis):
        """Set axis identifer of master to 'axis'.
        @param axis : Axis as string.
        """
        debug('GCSCommands.SCH(axis=%r)', axis)
        checksize((1,), axis)
        cmdstr = self.getcmdstr('SCH', axis)
        self.__msgs.send(cmdstr)

    def STP(self, noraise=False):
        """Stop all axes abruptly.
        Stop all motion caused by move commands (e.g. MOV, MVR, GOH, STE, SVA, SVR), referencing
        commands (e.g. FNL, FPL FRF), macros (e.g. MAC), wave generator output (e.g. WGO) and by
        the autozero procedure (e.g. ATZ) and by the user profile mode (e.g. UP*). Analog input is
        unconnected from the axes. Joystick is disabled.
        May raise GCSError(E10_PI_CNTR_STOP).
        @param noraise : If True a GCS error 10 (controller was stopped by command) will not be raised.
        """
        debug('GCSCommands.STP()')
        cmdstr = self.getcmdstr('STP')
        try:
            self.__msgs.send(cmdstr)
        except pipython.GCSError as exc:
            if noraise and exc == pipython.gcserror.E10_PI_CNTR_STOP:
                debug('GCS error 10 is masked')
            else:
                raise

    def TWC(self):
        """Clear all triggers for the wave generator."""
        debug('GCSCommands.TWC()')
        cmdstr = self.getcmdstr('TWC')
        self.__msgs.send(cmdstr)

    def WGR(self):
        """Start a recording synchronized with the wave generator."""
        debug('GCSCommands.WGR()')
        cmdstr = self.getcmdstr('WGR')
        self.__msgs.send(cmdstr)

    def MAC_END(self):
        """End macro recording."""
        debug('GCSCommands.MAC_END()')
        cmdstr = self.getcmdstr('MAC END')
        self.__msgs.send(cmdstr)

    def StopAll(self, noraise=False):
        """Stop all axes abruptly by sending "#24".
        Stop all motion caused by move commands (e.g. MOV, MVR, GOH, STE, SVA, SVR), referencing
        commands (e.g. FNL, FPL FRF), macros (e.g. MAC), wave generator output (e.g. WGO) and by
        the autozero procedure (e.g. ATZ) and by the user profile mode (e.g. UP*). Analog input is
        unconnected from the axes. Joystick is disabled.
        May raise GCSError(E10_PI_CNTR_STOP).
        @param noraise : If True a GCS error 10 (controller was stopped by command) will not be raised.
        """
        debug('GCSCommands.StopAll(noraise=%s)', noraise)
        try:
            self.__msgs.send(chr(24))
        except pipython.GCSError as exc:
            if noraise and exc == pipython.gcserror.E10_PI_CNTR_STOP:
                debug('GCSError 10 is catched and will not raise')
            else:
                raise

    def SystemAbort(self):
        """Abort system, i.e. controller will halt or reboot."""
        debug('GCSCommands.SystemAbort()')
        self.__msgs.send(chr(27))

    def RTR(self, value):
        """Set the record table rate, i.e. the number of servo-loop cycles to be used in data
        recording operations (see DRC()). Settings larger than 1 make it possible to cover longer
        time periods with a limited number of points.
        @param value : Number of servo cycles as integer.
        """
        debug('GCSCommands.RTR(value=%r)', value)
        checksize((1,), value)
        cmdstr = self.getcmdstr('RTR', value)
        self.__msgs.send(cmdstr)

    def DEL(self, value):
        """Delay the command interpreter for 'value' milliseconds.
        DEL is used within macros primarily. Do not mistake MAC DEL which deletes macros for DEL which delays.
        This command can be interrupted with #24.
        @param value : Delay value in milliseconds as integer.
        """
        debug('GCSCommands.DEL(value=%r)', value)
        checksize((1,), value)
        cmdstr = self.getcmdstr('DEL', value)
        self.__msgs.send(cmdstr)

    def SAV(self, axes=None):
        """Save stage configuration and motion parameters to non-volatile memory.
        @param axes : Axes as string convertible or list of them.
        """
        debug('GCSCommands.SAV(axes=%r)', axes)
        cmdstr = self.getcmdstr('SAV', axes)
        self.__msgs.send(cmdstr)

    def FLM(self, axis, length, threshold=None, line=None, direction=None):
        """Fast line scan to maximum over specified 'length' along an 'axis'.
        This function does not stop when the threshold level is reached, but performs a complete
        scan of the scan line.
        @param axis : Name of scan axis as string.
        @param length : Scan length in mm or degree as float.
        @param threshold : Threshold level at analog input as float.
        @param line : Identifier of the analog input signal as integer.
        @param direction: As integer, 0: centered, 1: positive, 2 : negative.
        """
        debug('GCSCommands.FLM(axis=%r, length=%r, threshold=%r, line=%r, direction=%r)', axis, length,
              threshold, line, direction)
        checksize((1, 1), axis, length)
        cmdstr = self.getcmdstr('FLM', axis, length)
        cmdstr += '' if threshold is None else ' %s' % self.getcmdstr('L', threshold)
        cmdstr += '' if line is None else ' %s' % self.getcmdstr('A', line)
        cmdstr += '' if direction is None else ' %s' % self.getcmdstr('D', direction)
        self.__msgs.send(cmdstr)

    def FLS(self, axis, length, threshold=None, line=None, direction=None):
        """Fast line scan to maximum over specified 'length' along an 'axis'.
        The scan starts at the current position and returns to this position if the threshold level
        is not exceeded. If it is exceeded during the scan, the system stops at this position.
        @param axis : Name of scan axis as string.
        @param length : Scan length in mm or degree as float.
        @param threshold : Threshold level at analog input as float.
        @param line : Identifier of the analog input signal as integer.
        @param direction: As integer, 0: centered, 1: positive, 2 : negative.
        """
        debug('GCSCommands.FLS(axis=%r, length=%r, threshold=%r, line=%r, direction=%r)', axis, length,
              threshold, line, direction)
        checksize((1, 1), axis, length)
        cmdstr = self.getcmdstr('FLS', axis, length)
        cmdstr += '' if threshold is None else ' %s' % self.getcmdstr('L', threshold)
        cmdstr += '' if line is None else ' %s' % self.getcmdstr('A', line)
        cmdstr += '' if direction is None else ' %s' % self.getcmdstr('D', direction)
        self.__msgs.send(cmdstr)

    def ACC(self, axes, values=None):
        """Set the acceleration to use during moves of 'axes'.
        The setting only takes effect when the given axis is in closed-loop operation (servo on).
        @param axes: Axis or list of axes or dictionary {axis : value}.
        @param values : Float or list of floats or None.
        """
        debug('GCSCommands.ACC(axes=%r, values=%r)', axes, values)
        axes, values = getitemsvaluestuple(axes, values)
        cmdstr = self.getcmdstr('ACC', axes, values)
        self.__msgs.send(cmdstr)

    def ADD(self, varname, value1, value2):
        """Add 'value1' and 'value2' and save the result to variable 'varname'.
        @param varname : Name of target variable as string.
        @param value1 : Value to add as float.
        @param value2 : Value to add as float.
        """
        debug('GCSCommands.ADD(varname=%r, value1=%s, value2=%s)', varname, value1, value2)
        checksize((1, 1, 1), varname, value1, value2)
        cmdstr = self.getcmdstr('ADD', varname, value1, value2)
        self.__msgs.send(cmdstr)

    def DEC(self, axes, values=None):
        """Set the deceleration to use during moves of 'axes'.
        The setting only takes effect when the given axis is in closed-loop operation (servo on).
        @param axes: Axis or list of axes or dictionary {axis : value}.
        @param values : Float or list of floats or None.
        """
        debug('GCSCommands.DEC(axes=%r, values=%r)', axes, values)
        axes, values = getitemsvaluestuple(axes, values)
        cmdstr = self.getcmdstr('DEC', axes, values)
        self.__msgs.send(cmdstr)

    def DFF(self, axes, values=None):
        """Set the scale factor which is applied to the standard unit.
        E.g. 25.4 changes the physical unit from mm to inches.
        @param axes: Axis or list of axes or dictionary {axis : value}.
        @param values : Float or list of floats or None.
        """
        debug('GCSCommands.DFF(axes=%r, values=%r)', axes, values)
        axes, values = getitemsvaluestuple(axes, values)
        cmdstr = self.getcmdstr('DFF', axes, values)
        self.__msgs.send(cmdstr)

    def OAC(self, axes, values=None):
        """Set open-loop acceleration of 'axes'.
        The OAC setting only takes effect when the given axis is in open-loop operation (servo off).
        @param axes: Axis or list of axes or dictionary {axis : value}.
        @param values : Float or list of floats or None.
        """
        debug('GCSCommands.OAC(axes=%r, values=%r)', axes, values)
        axes, values = getitemsvaluestuple(axes, values)
        cmdstr = self.getcmdstr('OAC', axes, values)
        self.__msgs.send(cmdstr)

    def OAD(self, channels, values=None):
        """Get open-loop analog driving of the given PiezoWalk 'channels'.
        Servo must be disabled for the commanded axis prior to using this command
        (open-loop operation).
        @param channels: Axis or list of integers or dictionary {channel : value}.
        @param values : Float or list of floats or None.
        """
        debug('GCSCommands.OAD(channels=%r, values=%r)', channels, values)
        channels, values = getitemsvaluestuple(channels, values)
        cmdstr = self.getcmdstr('OAD', channels, values)
        self.__msgs.send(cmdstr)

    def ODC(self, channels, values=None):
        """Set open-loop deceleration of the given PiezoWalk 'channels'.
        The ODC setting only takes effect when the given axis is in open-loop operation (servo off).
        @param channels: Axis or list of integers or dictionary {channel : value}.
        @param values : Float or list of floats or None.
        """
        debug('GCSCommands.ODC(channels=%r, values=%r)', channels, values)
        channels, values = getitemsvaluestuple(channels, values)
        cmdstr = self.getcmdstr('ODC', channels, values)
        self.__msgs.send(cmdstr)

    def OSM(self, channels, values=None):
        """Get open-loop step moving of the given PiezoWalk 'channels'.
        Prior to using OSM, servo must be disabled for the axis to
        which the PiezoWalk channel is assigned (open-loop operation).
        @param channels: Axis or list of channels or dictionary {channel : value}.
        @param values : Float or integer convertible or list of them or None.
        """
        debug('GCSCommands.OSM(channels=%r, values=%r)', channels, values)
        channels, values = getitemsvaluestuple(channels, values)
        cmdstr = self.getcmdstr('OSM', channels, values)
        self.__msgs.send(cmdstr)

    def OVL(self, channels, values=None):
        """Set velocity for open-loop nanostepping motion.
        The OVL() setting only takes effect when the given axis is in
        open-loop operation (servo off).
        @param channels: Axis or list of integers or dictionary {channel : value}.
        @param values : Float or list of floats or None.
        """
        debug('GCSCommands.OVL(channels=%r, values=%r)', channels, values)
        channels, values = getitemsvaluestuple(channels, values)
        cmdstr = self.getcmdstr('OVL', channels, values)
        self.__msgs.send(cmdstr)

    def POS(self, axes, values=None):
        """Set current position for given 'axes' (does not cause motion).
        An axis is considered as "referenced" when the position was
        set with POS(), so that qFRF() replies True. Setting the current
        position with POS() is only possible when the referencing mode is set
        to False, see RON().
        @param axes: Axis or list of axes or dictionary {axis : value}.
        @param values : Float or list of floats or None.
        """
        debug('GCSCommands.POS(axes=%r, values=%r)', axes, values)
        axes, values = getitemsvaluestuple(axes, values)
        cmdstr = self.getcmdstr('POS', axes, values)
        self.__msgs.send(cmdstr)

    def qACC(self, axes=None):
        """Get the acceleration values of 'axes' for closed-loop operation.
        @param axes : String convertible or list of them or None.
        @return Ordered dictionary of {axis: value}, values are float.
        """
        debug('GCSCommands.qACC(axes=%r)', axes)
        cmdstr = self.getcmdstr('ACC?', axes)
        answer = self.__msgs.read(cmdstr)
        answerdict = getdict_oneitem(answer, axes, valueconv=(float,))
        debug('GCSCommands.qACC = %r', answerdict)
        return answerdict

    def qAOS(self, axes=None):
        """Get analog input offset.
        @param axes : String convertible or list of them or None.
        @return : Ordered dictionary of {axis: value}, values are float.
        """
        debug('GCSCommands.qAOS(axes=%r)', axes)
        cmdstr = self.getcmdstr('AOS?', axes)
        answer = self.__msgs.read(cmdstr)
        answerdict = getdict_oneitem(answer, axes, valueconv=(float,))
        debug('GCSCommands.qAOS = %r', answerdict)
        return answerdict

    def qCAV(self, axes=None):
        """Get the current value of the variable controlled by the selected control mode.
        @param axes : String convertible or list of them or None.
        @return : Ordered dictionary of {axis: value}, values are float.
        """
        debug('GCSCommands.qCAV(axes=%r)', axes)
        cmdstr = self.getcmdstr('CAV?', axes)
        answer = self.__msgs.read(cmdstr)
        answerdict = getdict_oneitem(answer, axes, valueconv=(float,))
        debug('GCSCommands.qCAV = %r', answerdict)
        return answerdict

    def qCCV(self, axes=None):
        """Get currently valid control value of given axis.
        @param axes : String convertible or list of them or None.
        @return : Ordered dictionary of {axis: value}, values are float.
        """
        debug('GCSCommands.qCCV(axes=%r)', axes)
        cmdstr = self.getcmdstr('CCV?', axes)
        answer = self.__msgs.read(cmdstr)
        answerdict = getdict_oneitem(answer, axes, valueconv=(float,))
        debug('GCSCommands.qCCV = %r', answerdict)
        return answerdict

    def qCMN(self, axes=None):
        """Get the minimum commandable closed-loop target.
        @param axes : String convertible or list of them or None.
        @return : Ordered dictionary of {axis: value}, values are float.
        """
        debug('GCSCommands.qCMN(axes=%r)', axes)
        cmdstr = self.getcmdstr('CMN?', axes)
        answer = self.__msgs.read(cmdstr)
        answerdict = getdict_oneitem(answer, axes, valueconv=(float,))
        debug('GCSCommands.qCMN = %r', answerdict)
        return answerdict

    def qCMX(self, axes=None):
        """Get the maximum commandable closed-loop target.
        @param axes : String convertible or list of them or None.
        @return : Ordered dictionary of {axis: value}, values are float.
        """
        debug('GCSCommands.qCMX(axes=%r)', axes)
        cmdstr = self.getcmdstr('CMX?', axes)
        answer = self.__msgs.read(cmdstr)
        answerdict = getdict_oneitem(answer, axes, valueconv=(float,))
        debug('GCSCommands.qCMX = %r', answerdict)
        return answerdict

    def qCOV(self, channels=None):
        """Get current openloop velocity.
        @param channels : Integer convertible or list of them or None.
        @return : Ordered dictionary of {channel: value}, values are float.
        """
        debug('GCSCommands.qCOV(channels=%r)', channels)
        cmdstr = self.getcmdstr('COV?', channels)
        answer = self.__msgs.read(cmdstr)
        answerdict = getdict_oneitem(answer, channels, valueconv=(float,), itemconv=int)
        debug('GCSCommands.qCOV = %r', answerdict)
        return answerdict

    def qATC(self, channels=None):
        """Get auto calibration settings of 'channels'.
        @param channels : Integer convertible or list of them or None.
        @return : Ordered dictionary of {channel: value}, values are int.
        """
        debug('GCSCommands.qATC(channels=%r)', channels)
        cmdstr = self.getcmdstr('ATC?', channels)
        answer = self.__msgs.read(cmdstr)
        answerdict = getdict_oneitem(answer, channels, valueconv=(int,), itemconv=int)
        debug('GCSCommands.qATC = %r', answerdict)
        return answerdict

    def qNAV(self, channels=None):
        """Get the number of readout values of the analog input used for averaging.
        @param channels : Integer convertible or list of them or None.
        @return : Ordered dictionary of {channel: value}, values are int.
        """
        debug('GCSCommands.qNAV(channels=%r)', channels)
        cmdstr = self.getcmdstr('NAV?', channels)
        answer = self.__msgs.read(cmdstr)
        answerdict = getdict_oneitem(answer, channels, valueconv=(int,), itemconv=int)
        debug('GCSCommands.qNAV = %r', answerdict)
        return answerdict

    def qTAD(self, channels=None):
        """Get the ADC value for the given input signal 'channels'.
        @param channels : Integer convertible or list of them or None.
        @return : Ordered dictionary of {channel: value}, values are int.
        """
        debug('GCSCommands.qTAD(channels=%r)', channels)
        cmdstr = self.getcmdstr('TAD?', channels)
        answer = self.__msgs.read(cmdstr)
        answerdict = getdict_oneitem(answer, channels, valueconv=(int,), itemconv=int)
        debug('GCSCommands.qTAD = %r', answerdict)
        return answerdict

    def qTAV(self, channels=None):
        """Get the voltage value for the specified analog input 'channels'.
        @param channels : Integer convertible or list of them or None.
        @return : Ordered dictionary of {channel: value}, values are float.
        """
        debug('GCSCommands.qTAV(channels=%r)', channels)
        cmdstr = self.getcmdstr('TAV?', channels)
        answer = self.__msgs.read(cmdstr)
        answerdict = getdict_oneitem(answer, channels, valueconv=(float,), itemconv=int)
        debug('GCSCommands.qTAV = %r', answerdict)
        return answerdict

    def qTNS(self, channels=None):
        """Get the normalized value for the specified input signal 'channels'.
        @param channels : Integer convertible or list of them or None.
        @return : Ordered dictionary of {channel: value}, values are float.
        """
        debug('GCSCommands.qTNS(channels=%r)', channels)
        cmdstr = self.getcmdstr('TNS?', channels)
        answer = self.__msgs.read(cmdstr)
        answerdict = getdict_oneitem(answer, channels, valueconv=(float,), itemconv=int)
        debug('GCSCommands.qTNS = %r', answerdict)
        return answerdict

    def qTSP(self, channels=None):
        """Get the current position of the given input signal 'channels' in physical units.
        @param channels : Integer convertible or list of them or None.
        @return : Ordered dictionary of {channel: value}, values are float.
        """
        debug('GCSCommands.qTSP(channels=%r)', channels)
        cmdstr = self.getcmdstr('TSP?', channels)
        answer = self.__msgs.read(cmdstr)
        answerdict = getdict_oneitem(answer, channels, valueconv=(float,), itemconv=int)
        debug('GCSCommands.qTSP = %r', answerdict)
        return answerdict

    def qVOL(self, channels=None):
        """Get the current piezo voltages for 'channels'.
        @param channels : Integer convertible or list of them or None.
        @return : Ordered dictionary of {channel: value}, values are float.
        """
        debug('GCSCommands.qVOL(channels=%r)', channels)
        cmdstr = self.getcmdstr('VOL?', channels)
        answer = self.__msgs.read(cmdstr)
        answerdict = getdict_oneitem(answer, channels, valueconv=(float,), itemconv=int)
        debug('GCSCommands.qVOL = %r', answerdict)
        return answerdict

    def qSGA(self, channels=None):
        """Get the gain value for the given analog input 'channels'.
        @param channels : Integer convertible or list of them or None.
        @return : Ordered dictionary of {channel: value}, values are int.
        """
        debug('GCSCommands.qSGA(channels=%r)', channels)
        cmdstr = self.getcmdstr('SGA?', channels)
        answer = self.__msgs.read(cmdstr)
        answerdict = getdict_oneitem(answer, channels, valueconv=(int,), itemconv=int)
        debug('GCSCommands.qSGA = %r', answerdict)
        return answerdict

    def qDEC(self, axes=None):
        """Get the deceleration value for closed-loop operation of 'axes'.
        @param axes : String convertible or list of them or None.
        @return : Ordered dictionary of {axis: value}, values are float.
        """
        debug('GCSCommands.qDEC(axes=%r)', axes)
        cmdstr = self.getcmdstr('DEC?', axes)
        answer = self.__msgs.read(cmdstr)
        answerdict = getdict_oneitem(answer, axes, valueconv=(float,))
        debug('GCSCommands.qDEC = %r', answerdict)
        return answerdict

    def qFPH(self, axes=None):
        """Get found phase.
        Offset between motor and encoder. No motion is started.
        An answer "-1" means that the phase has not yet been determined. Run FPH to do so.
        @param axes : String convertible or list of them or None.
        @return : Ordered dictionary of {axis: value}, values are float.
        """
        debug('GCSCommands.qFPH(axes=%r)', axes)
        cmdstr = self.getcmdstr('FPH?', axes)
        answer = self.__msgs.read(cmdstr)
        answerdict = getdict_oneitem(answer, axes, valueconv=(float,))
        debug('GCSCommands.qFPH = %r', answerdict)
        return answerdict

    def qDFF(self, axes=None):
        """Get scale factors for 'axes' set with DFF().
        @param axes : String convertible or list of them or None.
        @return : Ordered dictionary of {axis: value}, values are float.
        """
        debug('GCSCommands.qDFF(axes=%r)', axes)
        cmdstr = self.getcmdstr('DFF?', axes)
        answer = self.__msgs.read(cmdstr)
        answerdict = getdict_oneitem(answer, axes, valueconv=(float,))
        debug('GCSCommands.qDFF = %r', answerdict)
        return answerdict

    def qDFH(self, axes=None):
        """Get the sensor positions of the current home position definitions.
        @param axes : String convertible or list of them or None.
        @return : Ordered dictionary of {axis: value}, values are float.
        """
        debug('GCSCommands.qDFH(axes=%r)', axes)
        cmdstr = self.getcmdstr('DFH?', axes)
        answer = self.__msgs.read(cmdstr)
        answerdict = getdict_oneitem(answer, axes, valueconv=(float,))
        debug('GCSCommands.qDFH = %r', answerdict)
        return answerdict

    def qMOV(self, axes=None):
        """Get the commanded target positions for 'axes'.
        Use qPOS() to get the current positions.
        @param axes : String convertible or list of them or None.
        @return : Ordered dictionary of {axis: value}, values are float.
        """
        debug('GCSCommands.qMOV(axes=%r)', axes)
        cmdstr = self.getcmdstr('MOV?', axes)
        answer = self.__msgs.read(cmdstr)
        answerdict = getdict_oneitem(answer, axes, valueconv=(float,))
        debug('GCSCommands.qMOV = %r', answerdict)
        return answerdict

    def qJOG(self, axes=None):
        """Get the velocity and direction for motion caused by JOG().
        @param axes : String convertible or list of them or None.
        @return : Ordered dictionary of {axis: value}, values are float.
        """
        debug('GCSCommands.qJOG(axes=%r)', axes)
        cmdstr = self.getcmdstr('JOG?', axes)
        answer = self.__msgs.read(cmdstr)
        answerdict = getdict_oneitem(answer, axes, valueconv=(float,))
        debug('GCSCommands.qJOG = %r', answerdict)
        return answerdict

    def qNLM(self, axes=None):
        """Get lower limits ("soft limits") for the positions of 'axes'.
        @param axes : String convertible or list of them or None.
        @return : Ordered dictionary of {axis: value}, values are float.
        """
        debug('GCSCommands.qNLM(axes=%r)', axes)
        cmdstr = self.getcmdstr('NLM?', axes)
        answer = self.__msgs.read(cmdstr)
        answerdict = getdict_oneitem(answer, axes, valueconv=(float,))
        debug('GCSCommands.qNLM = %r', answerdict)
        return answerdict

    def qOAC(self, channels=None):
        """Get the current value of the open-loop acceleration of given 'channels'.
        @param channels : Integer convertible or list of them or None.
        @return : Ordered dictionary of {axis: value}, values are float.
        """
        debug('GCSCommands.qOAC(channels=%r)', channels)
        cmdstr = self.getcmdstr('OAC?', channels)
        answer = self.__msgs.read(cmdstr)
        answerdict = getdict_oneitem(answer, channels, valueconv=(float,))
        debug('GCSCommands.qOAC = %r', answerdict)
        return answerdict

    def qOAD(self, channels=None):
        """Get last commanded open loop analog driving voltage of given Nexline 'channels'.
        @param channels : Integer convertible or list of them or None.
        @return : Ordered dictionary of {axis: value}, values are float.
        """
        debug('GCSCommands.qOAD(channels=%r)', channels)
        cmdstr = self.getcmdstr('OAD?', channels)
        answer = self.__msgs.read(cmdstr)
        answerdict = getdict_oneitem(answer, channels, valueconv=(float,), itemconv=int)
        debug('GCSCommands.qOAD = %r', answerdict)
        return answerdict

    def qOCD(self, channels=None):
        """Get last commanded open loop clamp driving voltage of given Nexline 'channels'.
        @param channels : Integer convertible or list of them or None.
        @return : Ordered dictionary of {axis: value}, values are float.
        """
        debug('GCSCommands.qOCD(channels=%r)', channels)
        cmdstr = self.getcmdstr('OCD?', channels)
        answer = self.__msgs.read(cmdstr)
        answerdict = getdict_oneitem(answer, channels, valueconv=(float,), itemconv=int)
        debug('GCSCommands.qOCD = %r', answerdict)
        return answerdict

    def qDRR(self, tables=None, offset=None, numvalues=None):
        """Get data record 'tables'.
        Function returns the header data only. Use "while self.bufstate is not True" and then
        call self.bufdata to get the data. (see docs)
        This function reads the data asynchronously, it will return as soon as the data header has
        been read and start a background process which reads in the data itself.
        @param tables : Data recorder table ID as integer convertible or list of them or None.
        @param offset : Start point in the table as integer, starts with index 1.
        Required if 'numvalues' or 'tables' is given.
        @param numvalues : Number of points to be read per table as integer.
        Required if 'offset' or 'tables' is given.
        @return : Header as ordered dictionary.
        """
        debug('GCSCommands.qDRR(tables=%r, offset=%r, numvalues=%r)', tables, offset, numvalues)
        if offset is not None or numvalues is not None:
            checksize((1, 1), offset, numvalues)
        if tables:
            checksize((1, 1), offset, numvalues)
        cmdstr = self.getcmdstr('DRR?', offset, numvalues, tables)
        answer = self.__msgs.read(cmdstr, gcsdata=numvalues)
        answer = getgcsheader(answer)
        debug('GCSCommands.qDRR = %r', answer)
        return answer

    def qGFR(self, tables=None, offset=None, numvalues=None):
        """Get the results of last frequency response measurement.
        Function returns the header data only. Use "while self.bufstate is not True" and then
        call self.bufdata to get the data. (see docs)
        This function reads the data asynchronously, it will return as soon as the data header has
        been read and start a background process which reads in the data itself.
        @param tables : Data recorder table ID as integer convertible or list of them or None.
        @param offset : Start point in the table as integer, starts with index 1.
        Required if 'numvalues' or 'tables' is given.
        @param numvalues : Number of points to be read per table as integer.
        Required if 'offset' or 'tables' is given.
        @return : Header as ordered dictionary.
        """
        debug('GCSCommands.qGFR(tables=%r, offset=%r, numvalues=%r)', tables, offset, numvalues)
        if offset is not None or numvalues is not None:
            checksize((1, 1), offset, numvalues)
        if tables:
            checksize((1, 1), offset, numvalues)
        cmdstr = self.getcmdstr('GFR?', offset, numvalues, tables)
        answer = self.__msgs.read(cmdstr, gcsdata=numvalues)
        answer = getgcsheader(answer)
        debug('GCSCommands.qGFR = %r', answer)
        return answer

    def qDDL(self, tables=None, offset=None, numvalues=None):
        """Get dynamic linearization data of 'tables'.
        Function returns the header data only. Use "while self.bufstate is not True" and then
        call self.bufdata to get the data. (see docs)
        Generally the DDL tables don't have a common length. Note that the definition of the
        GCS array output doesn't allow reading of tables with different length at the same time.
        Use DTL() to read the table length before reading the table data.
        @param tables : DDL table ID as integer convertible or list of them or None.
        @param offset : Start point in the table as integer, starts with index 1.
        Required if 'numvalues' or 'tables' is given.
        @param numvalues : Number of points to be read per table as integer.
        Required if 'offset' or 'tables' is given.
        @return : Header as ordered dictionary.
        """
        debug('GCSCommands.qDDL(tables=%r, offset=%r, numvalues=%r)', tables, offset, numvalues)
        if offset is not None or numvalues is not None:
            checksize((1, 1), offset, numvalues)
        if tables:
            checksize((1, 1), offset, numvalues)
        cmdstr = self.getcmdstr('DDL?', offset, numvalues, tables)
        answer = self.__msgs.read(cmdstr, gcsdata=numvalues)
        answer = getgcsheader(answer)
        debug('GCSCommands.qDDL = %r', answer)
        return answer

    def qGWD(self, tables=None, offset=None, numvalues=None):
        """Get the waveform associated with wave 'tables'.
        Function returns the header data only. Use "while self.bufstate is not True" and then
        call self.bufdata to get the data. (see docs)
        Generally the wave tables don't have a common length. Note that the definition of the
        GCS array output doesn't allow reading of tables with different length at the same time.
        @param tables : Wave table ID as integer convertible or list of them or None.
        @param offset : Start point in the table as integer, starts with index 1.
        Required if 'numvalues' or 'tables' is given.
        @param numvalues : Number of points to be read per table as integer.
        Required if 'offset' or 'tables' is given.
        @return : Header as ordered dictionary.
        """
        debug('GCSCommands.qGWD(tables=%r, offset=%r, numvalues=%r)', tables, offset, numvalues)
        if offset is not None or numvalues is not None:
            checksize((1, 1), offset, numvalues)
        if tables:
            checksize((1, 1), offset, numvalues)
        cmdstr = self.getcmdstr('GWD?', offset, numvalues, tables)
        answer = self.__msgs.read(cmdstr, gcsdata=numvalues)
        answer = getgcsheader(answer)
        debug('GCSCommands.qGWD = %r', answer)
        return answer

    def qHIT(self, tables=None, offset=None, numvalues=None):
        """Get the human interface device lookup 'tables'.
        Function returns the header data only. Use "while self.bufstate is not True" and then
        call self.bufdata to get the data. (see docs)
        @param tables : Lookup table ID as integer convertible or list of them or None.
        @param offset : Start point in the table as integer, starts with index 1.
        Required if 'numvalues' or 'tables' is given.
        @param numvalues : Number of points to be read per table as integer.
        Required if 'offset' or 'tables' is given.
        @return : Header as ordered dictionary.
        """
        debug('GCSCommands.qHIT(tables=%r, offset=%r, numvalues=%r)', tables, offset, numvalues)
        if offset is not None or numvalues is not None:
            checksize((1, 1), offset, numvalues)
        if tables:
            checksize((1, 1), offset, numvalues)
        cmdstr = self.getcmdstr('HIT?', offset, numvalues, tables)
        answer = self.__msgs.read(cmdstr, gcsdata=numvalues)
        answer = getgcsheader(answer)
        debug('GCSCommands.qHIT = %r', answer)
        return answer

    def qJLT(self, offset=None, numvalues=None, devices=None, devaxes=None):
        """Get the joystick lookup 'tables'.
        Function returns the header data only. Use "while self.bufstate is not True" and then
        call self.bufdata to get the data. (see docs)
        @param offset : Start point in the table as integer, starts with index 1.
        Required if 'numvalues' or 'devices'/'devaxes' is given.
        @param numvalues : Number of points to be read per table as integer.
        Required if 'devices'/'devaxes' is given.
        @param devices: Device ID or list of devices or None.
        @param devaxes : Integer convertible or list of them or None.
        @return : Header as ordered dictionary.
        """
        debug('GCSCommands.qJLT(offset=%r, numvalues=%r, devices=%s, devaxes=%s)', offset, numvalues, devices, devaxes)
        if numvalues is not None:
            checksize((1, 1), offset, numvalues)
        if devices is not None or devaxes is not None:
            checksize((1, 1, True, True), offset, numvalues, devices, devaxes)
        cmdstr = self.getcmdstr('JLT?', offset, numvalues, devices, devaxes)
        answer = self.__msgs.read(cmdstr, gcsdata=numvalues)
        answer = getgcsheader(answer)
        debug('GCSCommands.qJLT = %r', answer)
        return answer

    def qTWS(self, start=None, count=None, trigout=None):
        """Get trigger points.
        Function returns the header data only. Use "while self.bufstate is not True" and then
        call self.bufdata to get the data. (see docs)
        @param start : Number of first trigger point, starts with index 1. Required if 'count' or 'trigout' is given.
        @param count : Count of trigger points as integer. Required if 'trigout' is given.
        @param trigout : Trigger output line as integer convertible or list of them or None.
        @return : Header as ordered dictionary.
        """
        debug('GCSCommands.qTWS(start=%r, count=%r, trigout=%r)', start, count, trigout)
        if start is not None or trigout is not None or count is not None:
            checksize((1,), start)
        if trigout is not None or count is not None:
            checksize((1,), count)
        cmdstr = self.getcmdstr('TWS?', start, count, trigout)
        answer = self.__msgs.read(cmdstr, gcsdata=count)
        answer = getgcsheader(answer)
        debug('GCSCommands.qTWS = %r', answer)
        return answer

    def qODC(self, channels=None):
        """Get current open-loop deceleration of the PiezoWalk 'channels'.
        @param channels : Integer convertible or list of them or None.
        @return : Ordered dictionary of {axis: value}, values are float.
        """
        debug('GCSCommands.qODC(channels=%r)', channels)
        cmdstr = self.getcmdstr('ODC?', channels)
        answer = self.__msgs.read(cmdstr)
        answerdict = getdict_oneitem(answer, channels, valueconv=(float,))
        debug('GCSCommands.qODC = %r', answerdict)
        return answerdict

    def qOSM(self, channels=None):
        """Get the number of steps set by last OSM command for the given Nexline 'channels'.
        @param channels : Integer or float convertible or list of them or None.
        @return : Ordered dictionary of {axis: value}, values are float.
        """
        debug('GCSCommands.qOSM(channels=%r)', channels)
        cmdstr = self.getcmdstr('OSM?', channels)
        answer = self.__msgs.read(cmdstr)
        answerdict = getdict_oneitem(answer, channels, valueconv=(float,), itemconv=int)
        debug('GCSCommands.qOSM = %r', answerdict)
        return answerdict

    def qOVL(self, channels=None):
        """Get the current velocity for open-loop nanostepping motion of given 'channels'.
        @param channels : Integer convertible or list of them or None.
        @return : Ordered dictionary of {axis: value}, values are float.
        """
        debug('GCSCommands.qOVL(channels=%r)', channels)
        cmdstr = self.getcmdstr('OVL?', channels)
        answer = self.__msgs.read(cmdstr)
        answerdict = getdict_oneitem(answer, channels, valueconv=(float,), itemconv=int)
        debug('GCSCommands.qOVL = %r', answerdict)
        return answerdict

    def qPLM(self, axes=None):
        """Get upper limits ("soft limits") for the positions of 'axes'.
        @param axes : String convertible or list of them or None.
        @return : Ordered dictionary of {axis: value}, values are float.
        """
        debug('GCSCommands.qPLM(axes=%r)', axes)
        cmdstr = self.getcmdstr('PLM?', axes)
        answer = self.__msgs.read(cmdstr)
        answerdict = getdict_oneitem(answer, axes, valueconv=(float,))
        debug('GCSCommands.qPLM = %r', answerdict)
        return answerdict

    def qSPI(self, axes=None):
        """Get the pivot point coordinates for 'axes' in the volatile memory.
        @param axes : String convertible or list of them or None.
        @return : Ordered dictionary of {axis: value}, values are float.
        """
        debug('GCSCommands.qSPI(axes=%r)', axes)
        cmdstr = self.getcmdstr('SPI?', axes)
        answer = self.__msgs.read(cmdstr)
        answerdict = getdict_oneitem(answer, axes, valueconv=(float,))
        debug('GCSCommands.qSPI = %r', answerdict)
        return answerdict

    def qSSA(self, channels=None):
        """Get the current voltage amplitude for nanostepping motion of given 'channels'.
        @param channels : Integer convertible or list of them or None.
        @return : Ordered dictionary of {axis: value}, values are float.
        """
        debug('GCSCommands.qSSA(channels=%r)', channels)
        cmdstr = self.getcmdstr('SSA?', channels)
        answer = self.__msgs.read(cmdstr)
        answerdict = getdict_oneitem(answer, channels, valueconv=(float,), itemconv=int)
        debug('GCSCommands.qSSA = %r', answerdict)
        return answerdict

    def qSST(self, axes=None):
        """Get the distance ("step size") for motions.
        of the given 'axes' that are triggered by a manual control unit.
        @param axes : String convertible or list of them or None.
        @return : Ordered dictionary of {axis: value}, values are float.
        """
        debug('GCSCommands.qSST(axes=%r)', axes)
        cmdstr = self.getcmdstr('SST?', axes)
        answer = self.__msgs.read(cmdstr)
        answerdict = getdict_oneitem(answer, axes, valueconv=(float,))
        debug('GCSCommands.qSST = %r', answerdict)
        return answerdict

    def qSVA(self, axes=None):
        """Get the last valid open-loop control value for 'axes'.
        @param axes : String convertible or list of them or None.
        @return : Ordered dictionary of {axis: value}, values are float.
        """
        debug('GCSCommands.qSVA(axes=%r)', axes)
        cmdstr = self.getcmdstr('SVA?', axes)
        answer = self.__msgs.read(cmdstr)
        answerdict = getdict_oneitem(answer, axes, valueconv=(float,))
        debug('GCSCommands.qSVA = %r', answerdict)
        return answerdict

    def qTCV(self, axes=None):
        """Get the current value of the velocity for closed-loop operation.
        (value calculated by the profile generator) for 'axes'.
        @param axes : String convertible or list of them or None.
        @return : Ordered dictionary of {axis: value}, values are float.
        """
        debug('GCSCommands.qTCV(axes=%r)', axes)
        cmdstr = self.getcmdstr('TCV?', axes)
        answer = self.__msgs.read(cmdstr)
        answerdict = getdict_oneitem(answer, axes, valueconv=(float,))
        debug('GCSCommands.qTCV = %r', answerdict)
        return answerdict

    def qTMN(self, axes=None):
        """Get the low end of the travel range of 'axes'.
        @param axes : String convertible or list of them or None.
        @return : Ordered dictionary of {axis: value}, values are float.
        """
        debug('GCSCommands.qTMN(axes=%r)', axes)
        cmdstr = self.getcmdstr('TMN?', axes)
        answer = self.__msgs.read(cmdstr)
        answerdict = getdict_oneitem(answer, axes, valueconv=(float,))
        debug('GCSCommands.qTMN = %r', answerdict)
        return answerdict

    def qTMX(self, axes=None):
        """Get the high end of the travel range of 'axes'.
        @param axes : String convertible or list of them or None.
        @return : Ordered dictionary of {axis: value}, values are float.
        """
        debug('GCSCommands.qTMX(axes=%r)', axes)
        cmdstr = self.getcmdstr('TMX?', axes)
        answer = self.__msgs.read(cmdstr)
        answerdict = getdict_oneitem(answer, axes, valueconv=(float,))
        debug('GCSCommands.qTMX = %r', answerdict)
        return answerdict

    def qVEL(self, axes=None):
        """Get the velocity value commanded with VEL() for 'axes'.
        @param axes : String convertible or list of them or None.
        @return : Ordered dictionary of {axis: value}, values are float.
        """
        debug('GCSCommands.qVEL(axes=%r)', axes)
        cmdstr = self.getcmdstr('VEL?', axes)
        answer = self.__msgs.read(cmdstr)
        answerdict = getdict_oneitem(answer, axes, valueconv=(float,))
        debug('GCSCommands.qVEL = %r', answerdict)
        return answerdict

    def qVMA(self, channels=None):
        """Get upper PZT voltage soft limit of given piezo channels as 'channels'.
        @param channels : String convertible or list of them or None.
        @return : Ordered dictionary of {channel: value}, values are float.
        """
        debug('GCSCommands.qVMA(channels=%r)', channels)
        cmdstr = self.getcmdstr('VMA?', channels)
        answer = self.__msgs.read(cmdstr)
        answerdict = getdict_oneitem(answer, channels, valueconv=(float,), itemconv=int)
        debug('GCSCommands.qVMA = %r', answerdict)
        return answerdict

    def qVMI(self, channels=None):
        """Get lower PZT voltage soft limit of given piezo channels as 'channels'.
        @param channels : can be list or single string or be omitted
        @return : Ordered dictionary of {channel: value}, values are float.
        """
        debug('GCSCommands.qVMI(channels=%r)', channels)
        cmdstr = self.getcmdstr('VMI?', channels)
        answer = self.__msgs.read(cmdstr)
        answerdict = getdict_oneitem(answer, channels, valueconv=(float,), itemconv=int)
        debug('GCSCommands.qVMI = %r', answerdict)
        return answerdict

    def qWOS(self, wavegens=None):
        """Get wave generator output offset of given 'wavegens'
        @param wavegens : Integer convertible or list of them or None.
        @return : Ordered dictionary of {axis: value}, values are float.
        """
        debug('GCSCommands.qWOS(wavegens=%r)', wavegens)
        cmdstr = self.getcmdstr('WOS?', wavegens)
        answer = self.__msgs.read(cmdstr)
        answerdict = getdict_oneitem(answer, wavegens, valueconv=(float,), itemconv=int)
        debug('GCSCommands.qWOS = %r', answerdict)
        return answerdict

    def RNP(self, channels, values=None):
        """Relax the piezos of given piezowalk 'channels' without motion.
        @param channels: Axis or list of integers or dictionary {channel : value}.
        @param values : Float or list of floats or None.
        """
        debug('GCSCommands.RNP(channels=%r, values=%r)', channels, values)
        channels, values = getitemsvaluestuple(channels, values)
        cmdstr = self.getcmdstr('RNP', channels, values)
        self.__msgs.send(cmdstr)

    def SSA(self, channels, values=None):
        """Set the voltage amplitude for nanostepping motion of 'channels'.
        @param channels: Axis or list of integers or dictionary {channel : value}.
        @param values : Float or list of floats or None.
        """
        debug('GCSCommands.SSA(channels=%r, values=%r)', channels, values)
        channels, values = getitemsvaluestuple(channels, values)
        cmdstr = self.getcmdstr('SSA', channels, values)
        self.__msgs.send(cmdstr)

    def SST(self, axes, values=None):
        """Set the distance ("step size") for motions.
        of the given 'axes' that are triggered by a manual control unit.
        @param axes: Axis or list of axes or dictionary {axis : value}.
        @param values : Float or list of floats or None.
        """
        debug('GCSCommands.SST(axes=%r, values=%r)', axes, values)
        axes, values = getitemsvaluestuple(axes, values)
        cmdstr = self.getcmdstr('SST', axes, values)
        self.__msgs.send(cmdstr)

    def VEL(self, axes, values=None):
        """Set the velocities to use during moves of 'axes'.
        The setting only takes effect when the given axis is in closed-loop operation (servo on).
        @param axes: Axis or list of axes or dictionary {axis : value}.
        @param values : Float or list of floats or None.
        """
        debug('GCSCommands.VEL(axes=%r, values=%r)', axes, values)
        axes, values = getitemsvaluestuple(axes, values)
        cmdstr = self.getcmdstr('VEL', axes, values)
        self.__msgs.send(cmdstr)

    def WOS(self, wavegens, values=None):
        """Set offsets to the output of wave generator 'wavegens'.
        The current wave generator output is then created by adding
        the offset value to the current wave value: Generator Output = Offset +
        Current Wave Value. Do not confuse the output-offset value set with
        WOS with the offset settings done during the waveform creation with
        the WAV functions. While the WAV offset belongs to only one
        waveform, the WOS offset is added to all waveforms which are output
        by the given wave generator. Deleting wave table content with WCL
        has no effect on the offset settings for the wave generator output.
        @param wavegens: Axis or list of integers or dictionary {wavegen : value}.
        @param values : Float or list of floats or None.
        """
        debug('GCSCommands.WOS(wavegens=%r, values=%r)', wavegens, values)
        wavegens, values = getitemsvaluestuple(wavegens, values)
        cmdstr = self.getcmdstr('WOS', wavegens, values)
        self.__msgs.send(cmdstr)

    def VLS(self, value):
        """Set the velocity for the moving platform of the Hexapod.
        @param value : Value as float.
        """
        debug('GCSCommands.VLS(value=%r)', value)
        checksize((1,), value)
        cmdstr = self.getcmdstr('VLS', value)
        self.__msgs.send(cmdstr)

    def TIM(self, value=None):
        """Set the milliseconds timer to given 'value'.
        @param value : Value as float, without parameter the timer is reset to zero.
        """
        debug('GCSCommands.TIM(value=%r)', value)
        if value:
            checksize((1,), value)
        cmdstr = self.getcmdstr('TIM', value)
        self.__msgs.send(cmdstr)

    def SCT(self, value):
        """Set cycle time in milliseconds for trajectory generator.
        @param value : Value as float.
        """
        debug('GCSCommands.SCT(value=%r)', value)
        checksize((1,), value)
        cmdstr = self.getcmdstr('SCT T', value)
        self.__msgs.send(cmdstr)

    def AOS(self, axes, values=None):
        """Set offsets of the analog input for the given 'axes'.
        @param axes: Axis or list of axes or dictionary {axis : value}.
        @param values : Float or list of floats or None.
        """
        debug('GCSCommands.AOS(axes=%r, values=%r)', axes, values)
        axes, values = getitemsvaluestuple(axes, values)
        cmdstr = self.getcmdstr('AOS', axes, values)
        self.__msgs.send(cmdstr)

    def VOL(self, channels, values=None):
        """Set absolute PZT voltages for 'channels'.
        Servo must be switched off when calling this function.
        @param channels: Channel or list of channels or dictionary {channel : value}.
        @param values : Float or list of floats or None.
        """
        debug('GCSCommands.VOL(channels=%r, values=%r)', channels, values)
        channels, values = getitemsvaluestuple(channels, values)
        cmdstr = self.getcmdstr('VOL', channels, values)
        self.__msgs.send(cmdstr)

    def CST(self, axes, values=None):
        """Send the specific stage parameters for 'axes'. The property self.axes is reset.
        @param axes: Axis or list of axes or dictionary {axis : value} as string convertible.
        @param values : String convertible or list of them or None.
        """
        axes, values = getitemsvaluestuple(axes, values)
        cmdstr = self.getcmdstr('CST', axes, values)
        self.__msgs.send(cmdstr)
        del self.axes

    def CTR(self, axes, values=None):
        """Set target relative to current closed-loop target, moves the given axis.
        The control mode is to be selected via parameter (Closed-Loop Control Mode,
        ID 0x07030100) or via "CMO". Use "CMN?" and "CMX?" to get the currently valid limits.
        @param axes: Axis or list of axes or dictionary {axis : value}.
        @param values : Float or list of floats or None.
        """
        debug('GCSCommands.CTR(axes=%r, values=%r)', axes, values)
        axes, values = getitemsvaluestuple(axes, values)
        cmdstr = self.getcmdstr('CTR', axes, values)
        self.__msgs.send(cmdstr)

    def CTV(self, axes, values=None):
        """Set absolue closed-loop target, moves the given axis.
        The control mode is to be selected via parameter (Closed-Loop Control Mode,
        ID 0x07030100) or via "CMO". Use "CMN?" and "CMX?" to get the currently valid limits.
        @param axes: Axis or list of axes or dictionary {axis : value}.
        @param values : Float or list of floats or None.
        """
        debug('GCSCommands.CTV(axes=%r, values=%r)', axes, values)
        axes, values = getitemsvaluestuple(axes, values)
        cmdstr = self.getcmdstr('CTV', axes, values)
        self.__msgs.send(cmdstr)

    def DMOV(self, axes, values=None):
        """Move 'axes' to absolute positions.
        This command is very similar to the "MOV" command. During "MOV!"
        command motion, a new target position can be set by a new "MOV!" command.
        When this is done, the new targets will be programmed immediately.
        Motion will change in a smooth manner according to the acceleration
        limitation settings in the C842data.dat configuration file entry. By
        sending "MOV!" commands at periodic intervals, it is possible to avoid
        pauses in motion between moves. Furthermore, the "MOV!" command adjusts
        the velocity so as to reach the target at the end of the interval which
        is set using the "SCT" command.
        @param axes: Axis or list of axes or dictionary {axis : value}.
        @param values : Float or list of floats or None.
        """
        debug('GCSCommands.DMOV(axes=%r, values=%r)', axes, values)
        axes, values = getitemsvaluestuple(axes, values)
        cmdstr = self.getcmdstr('MOV!', axes, values)
        self.__msgs.send(cmdstr)

    def GetPosStatus(self, axes=None):
        """Get current position, corresponds to GCS command "#3" which behaves like "POS?".
        @param axes : String convertible or list of them or None.
        @return Ordered dictionary of {axis: value}, values are float.
        """
        debug('GCSCommands.GetPosStatus(axes=%r)', axes)
        checksize((), axes)
        answer = self.__msgs.read(chr(3))
        answerdict = getdict_oneitem(answer, axes, valueconv=(float,))
        debug('GCSCommands.GetPosStatus = %r', answerdict)
        return answerdict

    def MAS(self, axes, masters=None):
        """Set the electronic gearing master axes for 'axes'.
        @param axes: Item or list of axes or dictionary {axis : master} as string convertible.
        @param masters : String convertible or list of them or None.
        """
        debug('GCSCommands.MAS(axes=%r, masters=%r)', axes, masters)
        axes, masters = getitemsvaluestuple(axes, masters)
        cmdstr = self.getcmdstr('MAS', axes, masters)
        self.__msgs.send(cmdstr)

    def MOV(self, axes, values=None):
        """Move 'axes' to specified absolute positions.
        Axes will start moving to the new positions if ALL given targets are within the allowed
        ranges and ALL axes can move. All axes start moving simultaneously. Servo must be enabled
        for all 'axes' prior to using this command.
        @param axes: Axis or list of axes or dictionary {axis : value}.
        @param values : Float convertible or list of them or None.
        """
        debug('GCSCommands.MOV(axes=%r, values=%r)', axes, values)
        axes, values = getitemsvaluestuple(axes, values)
        cmdstr = self.getcmdstr('MOV', axes, values)
        self.__msgs.send(cmdstr)

    def PUN(self, axes, values=None):
        """Set the physical unit of 'axes' to 'values'.
        Setting does not affect the corresponding value of the axis. Command is CCL 1.
        @param axes: Axis or list of axes or dictionary {axis : value}.
        @param values : String or list of them or None.
        """
        debug('GCSCommands.PUN(axes=%r, values=%r)', axes, values)
        axes, values = getitemsvaluestuple(axes, values)
        cmdstr = self.getcmdstr('PUN', axes, values)
        self.__msgs.send(cmdstr)

    def TGA(self, trajectories, values=None):
        """Append 'values' to 'trajectories'.
        @param trajectories: Trajectory ID as int or list of them or dict {trajectory : value}.
        @param values : Float or list of them or None.
        """
        debug('GCSCommands.TGA(trajectories=%r, values=%r)', trajectories, values)
        trajectories, values = getitemsvaluestuple(trajectories, values)
        cmdstr = self.getcmdstr('TGA', trajectories, values)
        self.__msgs.send(cmdstr)

    def TGF(self, trajectories):
        """Finish 'trajectories'.
        Once a trajectory has been started you must continuously append new values to it to prevent
        a data buffer underrun. With TGF you signal the firmware that the trajectory will end soon.
        Hence no buffer underrun error will be set.
        @param trajectories: Trajectory ID as int or list of them, required.
        """
        debug('GCSCommands.TGF(trajectories=%r)', trajectories)
        checksize((True,), trajectories)
        cmdstr = self.getcmdstr('TGF', trajectories)
        self.__msgs.send(cmdstr)

    def TGT(self, value):
        """Set trajectory timing to 'value' which is unique for all trajectories.
        @param value: Trajectory timing in number of servo cycles as int.
        """
        debug('GCSCommands.TGT(value=%r)', value)
        checksize((1,), value)
        cmdstr = self.getcmdstr('TGT', value)
        self.__msgs.send(cmdstr)

    def qTGT(self):
        """Get trajectory timing which is unique for all trajectories.
        @return : Answer in number of servo cycles as integer.
        """
        debug('GCSCommands.qTGT()')
        answer = self.__msgs.read('TGT?')
        value = int(answer.strip())
        debug('GCSCommands.qTGT = %r', value)
        return value

    def qTWT(self):
        """Get the number of wave tables.
        @return : Answer as integer.
        """
        debug('GCSCommands.qTWT()')
        answer = self.__msgs.read('TWT?')
        value = int(answer.strip())
        debug('GCSCommands.qTWT = %r', value)
        return value

    def TGC(self, trajectories=None):
        """Clear 'trajectories'.
        Without argument all trajectories are cleared.
        @param trajectories: Trajectory ID as int or list of them or None.
        """
        debug('GCSCommands.TGC(trajectories=%r)', trajectories)
        cmdstr = self.getcmdstr('TGC', trajectories)
        self.__msgs.send(cmdstr)

    def TGS(self, trajectories=None):
        """Start 'trajectories'.
        Without argument all trajectories are started. On error all trajectories are stopped.
        @param trajectories: Trajectory ID as int or list of them or None.
        """
        debug('GCSCommands.TGS(trajectories=%r)', trajectories)
        cmdstr = self.getcmdstr('TGS', trajectories)
        self.__msgs.send(cmdstr)

    def qTGL(self, trajectories=None):
        """Get number of values stored in 'trajectories'.
        @param trajectories : Integer convertible or list of them or None.
        @return : Ordered dictionary of {trajectory: value}, values are int.
        """
        debug('GCSCommands.qTGL(trajectories=%r)', trajectories)
        cmdstr = self.getcmdstr('TGL?', trajectories)
        answer = self.__msgs.read(cmdstr)
        answerdict = getdict_oneitem(answer, trajectories, valueconv=(int,), itemconv=int)
        debug('GCSCommands.qTGL = %r', answerdict)
        return answerdict

    def ATC(self, channels, options=None):
        """Auto calibrate 'channels'.
        WPA 100 is required afterwards to store values in EEPROM. Command is CCL 1.
        @param channels: Channel or list of channels or dictionary {channel : value}.
        @param options : Option ID as integer, see controller manual.
        """
        debug('GCSCommands.ATC(channels=%r, options=%r)', channels, options)
        channels, options = getitemsvaluestuple(channels, options)
        cmdstr = self.getcmdstr('ATC', channels, options)
        self.__msgs.send(cmdstr)

    def JOG(self, axes, values=None):
        """Start motion with the given (constant) velocity for 'axes'.
        The sign of the velocity value gives the direction of motion. When motion started with
        JOG() is executed, the target value is changed continuously according to the given velocity.
        Motion started with JOG() is executed in addition to motion started with other move
        commands, e.g. MOV() or MVR().
        @param axes: Axis or list of axes or dictionary {axis : value}.
        @param values : Float convertible or list of them or None.
        """
        debug('GCSCommands.JOG(axes=%r, values=%r)', axes, values)
        axes, values = getitemsvaluestuple(axes, values)
        cmdstr = self.getcmdstr('JOG', axes, values)
        self.__msgs.send(cmdstr)

    def CMO(self, axes, values=None):
        """Set closed-loop control mode.
        The selection determines the controlled variable (e.g. position or velocity or force).
        @param axes: Axis or list of axes or dictionary {axis : value}.
        @param values : Integer convertible or list of them or None.
        """
        debug('GCSCommands.CMO(axes=%r, values=%r)', axes, values)
        axes, values = getitemsvaluestuple(axes, values)
        cmdstr = self.getcmdstr('CMO', axes, values)
        self.__msgs.send(cmdstr)

    def MVT(self, axes, values=None):
        """Set the "move triggered" mode ON or OFF.
        @param axes: Axis or list of axes or dictionary {axis : value}.
        @param values : Float convertible or list of them or None.
        """
        debug('GCSCommands.MVT(axes=%r, values=%r)', axes, values)
        axes, values = getitemsvaluestuple(axes, values)
        cmdstr = self.getcmdstr('MVT', axes, values)
        self.__msgs.send(cmdstr)

    def STE(self, axes, values=None):
        """Perform a step and record the step response for the given 'axes'.
        @param axes: Axis or list of axes or dictionary {axis : value}.
        @param values : Float convertible or list of them or None.
        """
        debug('GCSCommands.STE(axes=%r, values=%r)', axes, values)
        axes, values = getitemsvaluestuple(axes, values)
        cmdstr = self.getcmdstr('STE', axes, values)
        self.__msgs.send(cmdstr)

    def IMP(self, axes, values=None):
        """Perform an impulse and record the impulse response for the given 'axes'.
        @param axes: Axis or list of axes or dictionary {axis : value}.
        @param values : Float convertible or list of them or None.
        """
        debug('GCSCommands.IMP(axes=%r, values=%r)', axes, values)
        axes, values = getitemsvaluestuple(axes, values)
        cmdstr = self.getcmdstr('IMP', axes, values)
        self.__msgs.send(cmdstr)

    def AAP(self, axis1, length1, axis2, length2, stepsize=None, checks=None, line=None):
        """Start a scanning procedure for better determination of the maximum intensity.
        @param axis1 : First axis that defines a scanning area as string.
        @param length1 : Length of scanning area along 'axis1' or None.
        @param axis2 : Second axis that defines a scanning area as string.
        @param length2 : Length of scanning area along 'axis2' or None.
        @param stepsize : Starting value for the stepsize size as float.
        @param checks : Number of successful checks of the local maximum at the current position
        that is required for successfully completing as integer.
        @param line : Identifier of the analog input signal as integer.
        """
        debug('GCSCommands.AAP(axis1=%r, length1=%r, axis2=%r, length2=%r, stepsize=%r, checks=%r, line=%r)', axis1,
              length1, axis2, length2, stepsize, checks, line)
        checksize((1, 1, 1, 1), axis1, length1, axis2, length2)
        cmdstr = self.getcmdstr('AAP', axis1, length1, axis2, length2)
        cmdstr += '' if stepsize is None else ' %s' % self.getcmdstr('SA', stepsize)
        cmdstr += '' if checks is None else ' %s' % self.getcmdstr('N', checks)
        cmdstr += '' if line is None else ' %s' % self.getcmdstr('A', line)
        self.__msgs.send(cmdstr)

    def FIO(self, axis1, length1, axis2, length2, threshold=None, stepsize=None, angle=None, line=None):
        """Start a scanning procedure for alignment of optical elements.
        @param axis1 : First axis that defines a scanning area as string.
        @param length1 : Length of scanning area along 'axis1' or None.
        @param axis2 : Second axis that defines a scanning area as string.
        @param length2 : Length of scanning area along 'axis2' or None.
        @param threshold : Intensity threshold of the analog input signal in V as float.
        @param stepsize : Step size in which the platform moves along the spiral path as float.
        @param angle : Angle around the pivot point at which scanning is done in degrees as float.
        @param line : Identifier of the analog input signal as integer.
        """
        debug('GCSCommands.FIO(axis1=%r, length1=%r, axis2=%r, length2=%r, threshold=%r, stepsize=%r, angle=%r, '
              'line=%r)', axis1, length1, axis2, length2, threshold, stepsize, angle, line)
        checksize((1, 1, 1, 1), axis1, length1, axis2, length2)
        cmdstr = self.getcmdstr('FIO', axis1, length1, axis2, length2)
        cmdstr += '' if threshold is None else ' %s' % self.getcmdstr('S', threshold)
        cmdstr += '' if stepsize is None else ' %s' % self.getcmdstr('AR', stepsize)
        cmdstr += '' if angle is None else ' %s' % self.getcmdstr('L', angle)
        cmdstr += '' if line is None else ' %s' % self.getcmdstr('A', line)
        self.__msgs.send(cmdstr)

    def FSA(self, axis1, length1, axis2, length2, threshold=None, distance=None, stepsize=None, line=None):
        """Start a scanning procedure to determine the maximum intensity of an analog input signal
        in a plane. The search consists of a coarse and a fine portion.
        @param axis1 : First axis that defines a scanning area as string.
        @param length1 : Length of scanning area along 'axis1' or None.
        @param axis2 : Second axis that defines a scanning area as string.
        @param length2 : Length of scanning area along 'axis2' or None.
        @param threshold : Intensity threshold of the analog input signal in V as float.
        @param distance : Distance between the scanning lines during the coarse portion as float.
        @param stepsize : Step size in which the platform moves along the spiral path as float.
        @param line : Identifier of the analog input signal as integer.
        """
        debug('GCSCommands.FSA(axis1=%r, length1=%r, axis2=%r, length2=%r, threshold=%r, distance=%r, stepsize=%r, '
              'line=%r)', axis1, length1, axis2, length2, threshold, distance, stepsize, line)
        checksize((1, 1, 1, 1), axis1, length1, axis2, length2)
        cmdstr = self.getcmdstr('FSA', axis1, length1, axis2, length2)
        cmdstr += '' if threshold is None else ' %s' % self.getcmdstr('L', threshold)
        cmdstr += '' if distance is None else ' %s' % self.getcmdstr('S', distance)
        cmdstr += '' if stepsize is None else ' %s' % self.getcmdstr('SA', stepsize)
        cmdstr += '' if line is None else ' %s' % self.getcmdstr('A', line)
        self.__msgs.send(cmdstr)

    def FAA(self, axis, area, threshold, line):
        """Start the embedded fast scan "FAA - Fast Angular Line Scan to Maximum".
        @param axis : Axis to be used for the scan as string, must be U, V or W.
        @param area : Length of scan as float.
        @param threshold : Threshold level for analog input as float.
        @param line : Identifier of the analog input signal as integer.
        """
        debug('GCSCommands.FAA(axis=%r, area=%r, threshold=%r, line=%r)', axis, area, threshold, line)
        checksize((1, 1, 1, 1), axis, area, threshold, line)
        cmdstr = self.getcmdstr('FAA', axis, area, threshold, line)
        self.__msgs.send(cmdstr)

    def FAM(self, axes, area1, area2, threshold, stepsize, line):
        """Start the embedded fast scan "FAM - Fast Angular Scan to Maximum".
        @param axes : Axes to be used as string, must contain 2 valid rotation axis identifiers.
        @param area1 : Length of scan for the first axis as float.
        @param area2 : Length of scan for the second axis as float.
        @param threshold : Threshold level for analog input as float.
        @param stepsize : Step size for the scan as float.
        @param line : Identifier of the analog input signal as integer.
        """
        debug('GCSCommands.FAM(axes=%r, area1=%r, area2=%r, threshold=%r, stepsize=%r, line=%r)', axes, area1, area2,
              threshold, stepsize, line)
        checksize((1, 1, 1, 1, 1, 1), axes, area1, area2, threshold, stepsize, line)
        cmdstr = self.getcmdstr('FAM', axes, area1, area2, threshold, stepsize, line)
        self.__msgs.send(cmdstr)

    def WFR(self, axis, zeropoint, source, amplitude, lowfrq, highfrq, sweepsteps, sweepmode, veloffset):
        """Start an oscillation for a given frequency, amplitude, zero point and number of cycles and use the Goertzel
        algorithm to determine the system response (magnitude and phase shift) for this frequency.
        @param axis : Name of axis to be commanded as string.
        @param zeropoint : Zero point of oscillation as float .
        @param source : Source for analysis as int.
        @param amplitude : Amplitude of the oscillation as float.
        @param lowfrq : Lowest frequency of the oscillation as float.
        @param highfrq : Highest frequency of the oscillation as float.
        @param sweepsteps : Number of frequency steps during sweep as integer.
        @param sweepmode : Sweep mode, i.e. lin or log as integer.
        @param veloffset : Iis added to avoid stucking at reversal points as float.
        """
        debug('GCSCommands.FAM(axis=%r, zeropoint=%r, source=%r, amplitude=%r, lowfrq=%r, highfrq=%r, sweepsteps=%r, '
              'sweepmode=%r, veloffset=%r)', axis, zeropoint, source, amplitude, lowfrq, highfrq, sweepsteps, sweepmode,
              veloffset)
        checksize((1, 1, 1, 1, 1, 1, 1, 1, 1), axis, zeropoint, source, amplitude, lowfrq, highfrq, sweepsteps,
                  sweepmode, veloffset)
        cmdstr = self.getcmdstr('WFR', axis, zeropoint, source, amplitude, lowfrq, highfrq, sweepsteps, sweepmode,
                                veloffset)
        self.__msgs.send(cmdstr)

    def FAS(self, axes, area1, area2, threshold, stepsize, line):
        """Start the embedded fast scan "FAS - Fast Angular Scan".
        @param axes : Axes to be used as string, must contain 2 valid rotation axis identifiers.
        @param area1 : Length of scan for the first axis as float.
        @param area2 : Length of scan for the second axis as float.
        @param threshold : Threshold level for analog input as float.
        @param stepsize : Step size for the scan as float.
        @param line : Identifier of the analog input signal as integer.
        """
        debug('GCSCommands.FAS(axes=%r, area1=%r, area2=%r, threshold=%r, stepsize=%r, line=%r)', axes, area1, area2,
              threshold, stepsize, line)
        checksize((1, 1, 1, 1, 1, 1), axes, area1, area2, threshold, stepsize, line)
        cmdstr = self.getcmdstr('FAS', axes, area1, area2, threshold, stepsize, line)
        self.__msgs.send(cmdstr)

    def FSC(self, axis1, length1, axis2, length2, threshold=None, distance=None, line=None):
        """Start a scanning procedure which scans a specified area ("scanning area") until the
        analog input signal reaches a specified intensity threshold.
        @param axis1 : First axis that defines a scanning area as string.
        @param length1 : Length of scanning area along 'axis1' or None.
        @param axis2 : Second axis that defines a scanning area as string.
        @param length2 : Length of scanning area along 'axis2' or None.
        @param threshold : Intensity threshold of the analog input signal in V as float.
        @param distance : Distance between the scanning lines during the coarse portion as float.
        @param line : Identifier of the analog input signal as integer.
        """
        debug('GCSCommands.FSC(axis1=%r, length1=%r, axis2=%r, length2=%r, threshold=%r, distance=%r, line=%r)', axis1,
              length1, axis2, length2, threshold, distance, line)
        checksize((1, 1, 1, 1), axis1, length1, axis2, length2)
        cmdstr = self.getcmdstr('FSC', axis1, length1, axis2, length2)
        cmdstr += '' if threshold is None else ' %s' % self.getcmdstr('L', threshold)
        cmdstr += '' if distance is None else ' %s' % self.getcmdstr('S', distance)
        cmdstr += '' if line is None else ' %s' % self.getcmdstr('A', line)
        self.__msgs.send(cmdstr)

    def FSM(self, axis1, length1, axis2, length2, threshold=None, distance=None, line=None):
        """Start a scanning procedure to determine the global maximum intensity of an analog input
        signal in a plane.
        @param axis1 : First axis that defines a scanning area as string.
        @param length1 : Length of scanning area along 'axis1' or None.
        @param axis2 : Second axis that defines a scanning area as string.
        @param length2 : Length of scanning area along 'axis2' or None.
        @param threshold : Intensity threshold of the analog input signal in V as float.
        @param distance : Distance between the scanning lines during the coarse portion as float.
        @param line : Identifier of the analog input signal as integer.
        """
        debug('GCSCommands.FSM(axis1=%r, length1=%r, axis2=%r, length2=%r, threshold=%r, distance=%r, line=%r)', axis1,
              length1, axis2, length2, threshold, distance, line)
        checksize((1, 1, 1, 1), axis1, length1, axis2, length2)
        cmdstr = self.getcmdstr('FSM', axis1, length1, axis2, length2)
        cmdstr += '' if threshold is None else ' %s' % self.getcmdstr('L', threshold)
        cmdstr += '' if distance is None else ' %s' % self.getcmdstr('S', distance)
        cmdstr += '' if line is None else ' %s' % self.getcmdstr('A', line)
        self.__msgs.send(cmdstr)

    def WAV_LIN(self, table, firstpoint, numpoints, append, speedupdown, amplitude, offset, seglength):
        """Define a single scan line curve for given wave table.
        @param table : Wave table ID as integer.
        @param firstpoint : Index of the segment starting point in the wave table. Lowest possible value is 1.
        @param numpoints : Length of the single scan line curve as integer.
        @param append : "X" to start from first point, "&" to append and "+" to add to existing waveform.
        @param speedupdown : Number of points for speed up and down as integer.
        @param amplitude : Amplitude of the scan as float.
        @param offset : Offset of the scan as float.
        @param seglength : Length of the wave table segment as integer.
        """
        debug('GCSCommands.WAV_LIN(table=%r, firstpoint=%r, numpoints=%r, append=%r, speedupdown=%r, amplitude=%r, '
              'offset=%r, seglength=%r)', table, firstpoint, numpoints, append, speedupdown, amplitude, offset,
              seglength)
        checksize((1, 1, 1, 1, 1, 1, 1, 1), table, firstpoint, numpoints, append, speedupdown, amplitude, offset,
                  seglength)
        cmdstr = self.getcmdstr('WAV', table, append, 'LIN', seglength, amplitude, offset, numpoints, firstpoint,
                                speedupdown)
        self.__msgs.send(cmdstr)

    def WAV_SWEEP(self, table, append, startfreq, stopfreq, sweeptime, amplitude, offset):
        """Define a sweep scan for given wave table.
        @param table : Wave table ID as integer.
        @param append : "X" to start from first point, "&" to append and "+" to add to existing waveform.
        @param startfreq : Frequency at which to start the sweep in servorate/frequency as int. Example:
        servorate=25000Hz, desired start frequency=10Hz -> set to 25000/10=25000.
        @param stopfreq : Frequency at which to stop the sweep in servorate/frequency as int.
        @param sweeptime : Time for the sweep duration in servorate*sweeptime as int.
        @param amplitude : Amplitude of the sine sweep signal as float.
        @param offset : Offset of the sine sweep signal as float
        """
        debug('GCSCommands.WAV_SWEEP(table=%r, append=%r, startfreq=%r, stopfreq=%r, sweeptime=%r, amplitude=%r, '
              'offset=%r)', table, append, startfreq, stopfreq, sweeptime, amplitude, offset)
        checksize((1, 1, 1, 1, 1, 1, 1), table, append, startfreq, stopfreq, sweeptime, amplitude, offset)
        cmdstr = self.getcmdstr('WAV', table, append, 'SWEEP', startfreq, stopfreq, sweeptime, amplitude, offset)
        self.__msgs.send(cmdstr)

    def WAV_POL(self, table, append, firstpoint, numpoints, x0, a0, an):
        """Define a polynomial curve for given wave table.
        @param table : Wave table ID as integer.
        @param append : "X" to start from first point, "&" to append and "+" to add to existing waveform.
        @param firstpoint : Index of the segment starting point in the wave table. Lowest possible value is 1.
        @param numpoints : Length of the polynomial curve in points (cycle duration).
        @param x0 : Parameter as float in polynomial a0 + a1(x-x0)^1 + ... + an(x-x0)^n.
        @param a0 : Parameter as float in polynomial a0 + a1(x-x0)^1 + ... + an(x-x0)^n.
        @param an : Parameters as list of floats in polynomial a0 + a1(x-x0)^1 + ... + an(x-x0)^n.
        """
        debug('GCSCommands.WAV_POL(table=%r, append=%r, firstpoint=%r, numpoints=%r, x0=%r, a0=%r, '
              'an=%r)', table, append, firstpoint, numpoints, x0, a0, an)
        checksize((1, 1, 1, 1, 1, 1, True), table, append, firstpoint, numpoints, x0, a0, an)
        cmdstr = self.getcmdstr('WAV', table, append, 'POL', firstpoint, numpoints, x0, a0, an)
        self.__msgs.send(cmdstr)

    def WAV_SIN(self, table, append, firstpoint, numpoints, ampl, np, x0, phase, offset):
        """Define a sine curve ampl * sin (2pi / np * (x-x0) + phase) + offset for given wave table.
        @param table : Wave table ID as integer.
        @param append : "X" to start from first point, "&" to append x0d "+" to add to existing waveform.
        @param firstpoint : Index of the segment starting point in the wave table. Lowest possible value is 1.
        @param numpoints : Length of the single scan line curve as integer.
        @param ampl : Amplitude of the sine curve as float.
        @param np : Cycle duration of the sine curve in points as integer.
        @param x0 : Index of the starting point of the sine curve in the segment, lowest possible value is 0.
        @param phase : Phase of the sine curve as float.
        @param offset : Offset of the sine curve as float.
        """
        debug('GCSCommands.WAV_SIN(table=%r, append=%r, firstpoint=%r, numpoints=%r, ampl=%r, np=%r, '
              'x0=%r, phase=%s, offset=%s)', table, append, firstpoint, numpoints, ampl, np, x0, phase, offset)
        checksize((1, 1, 1, 1, 1, 1, 1, 1, 1), table, append, firstpoint, numpoints, ampl, np, x0, phase, offset)
        cmdstr = self.getcmdstr('WAV', table, append, 'SIN', firstpoint, numpoints, ampl, np, x0, phase, offset)
        self.__msgs.send(cmdstr)

    def WAV_TAN(self, table, append, firstpoint, numpoints, ampl, np, x0, phase, offset):
        """Define a sine curve ampl * tan (2pi / np * (x-x0) + phase) + offset for given wave table.
        @param table : Wave table ID as integer.
        @param append : "X" to start from first point, "&" to append x0d "+" to add to existing waveform.
        @param firstpoint : Index of the segment starting point in the wave table. Lowest possible value is 1.
        @param numpoints : Length of the single scan line curve as integer.
        @param ampl : Amplitude of the sine curve as float.
        @param np : Cycle duration of the sine curve in points as integer.
        @param x0 : Index of the starting point of the sine curve in the segment, lowest possible value is 0.
        @param phase : Phase of the sine curve as float.
        @param offset : Offset of the sine curve as float.
        """
        debug('GCSCommands.WAV_TAN(table=%r, append=%r, firstpoint=%r, numpoints=%r, ampl=%r, np=%r, '
              'x0=%r, phase=%s, offset=%s)', table, append, firstpoint, numpoints, ampl, np, x0, phase, offset)
        checksize((1, 1, 1, 1, 1, 1, 1, 1, 1), table, append, firstpoint, numpoints, ampl, np, x0, phase, offset)
        cmdstr = self.getcmdstr('WAV', table, append, 'TAN', firstpoint, numpoints, ampl, np, x0, phase, offset)
        self.__msgs.send(cmdstr)

    def WAV_RAMP(self, table, firstpoint, numpoints, append, center, speedupdown, amplitude, offset, seglength):
        """Define a ramp curve for given wave table.
        @param table : Wave table ID as integer.
        @param firstpoint : Index of the segment starting point in the wave table. Lowest possible value is 1.
        @param numpoints : Length of the single scan line curve as integer.
        @param append : "X" to start from first point, "&" to append and "+" to add to existing waveform.
        @param center : Index of the center point as integer, starts with 0.
        @param speedupdown : Number of points for speed up and down as integer.
        @param amplitude : Amplitude of the scan as float.
        @param offset : Offset of the scan as float.
        @param seglength : Length of the wave table segment as integer.
        """
        debug('GCSCommands.WAV_RAMP(table=%r, firstpoint=%r, numpoints=%r, append=%r, center=%r, speedupdown=%r, '
              'amplitude=%r, offset=%r, seglength=%r)', table, firstpoint, numpoints, append, center, speedupdown,
              amplitude, offset, seglength)
        checksize((1, 1, 1, 1, 1, 1, 1, 1, 1), table, firstpoint, numpoints, append, center, speedupdown, amplitude,
                  offset, seglength)
        cmdstr = self.getcmdstr('WAV', table, append, 'RAMP', seglength, amplitude, offset, numpoints, firstpoint,
                                speedupdown, center)
        self.__msgs.send(cmdstr)

    def WAV_SIN_P(self, table, firstpoint, numpoints, append, center, amplitude, offset, seglength):
        """Define a sine curve for given wave table.
        @param table : Wave table ID as integer.
        @param firstpoint : Index of the segment starting point in the wave table. Lowest possible value is 1.
        @param numpoints : Length of the single scan line curve as integer.
        @param append : "X" to start from first point, "&" to append and "+" to add to existing waveform.
        @param center : Index of the center point as integer, starts with 0.
        @param amplitude : Amplitude of the scan as float.
        @param offset : Offset of the scan as float.
        @param seglength : Length of the wave table segment as integer.
        """
        debug('GCSCommands.WAV_SIN_P(table=%r, firstpoint=%r, numpoints=%r, append=%r, center=%r, amplitude=%r, '
              'offset=%r, seglength=%r)', table, firstpoint, numpoints, append, center, amplitude, offset, seglength)
        checksize((1, 1, 1, 1, 1, 1, 1, 1), table, firstpoint, numpoints, append, center, amplitude, offset, seglength)
        cmdstr = self.getcmdstr('WAV', table, append, 'SIN_P', seglength, amplitude, offset, numpoints, firstpoint,
                                center)
        self.__msgs.send(cmdstr)

    def WAV_PNT(self, table, firstpoint, numpoints, append, wavepoint):
        """Define a user-defined curve for given 'table'. There will be no packaging hence all wavepoints will be sent
        as a single command. Use pitools.writewavepoints() to write the wavepoints in according bunches.
        @param table : Wave table ID as integer.
        @param firstpoint : Index of the segment starting point in the wave table. Lowest possible value is 1.
        @param numpoints : Length of the single scan line curve as integer.
        @param append : "X" to start from first point, "&" to append and "+" to add to existing waveform.
        @param wavepoint : Single wavepoint as float convertible or list of them.
        """
        debug('GCSCommands.WAV_PNT(table=%r, firstpoint=%r, numpoints=%r, append=%r, wavepoint=%r)', table, firstpoint,
              numpoints, append, wavepoint)
        checksize((1, 1, 1, 1, True), table, firstpoint, numpoints, append, wavepoint)
        cmdstr = self.getcmdstr('WAV', table, append, 'PNT', firstpoint, numpoints, wavepoint)
        self.__msgs.send(cmdstr)

    def CTO(self, lines, params, values):
        """Set trigger output conditions for the given digital output 'lines'.
        @param lines : Trigger output lines as integer convertible or list of them.
        @param params : Parameter IDs as integer convertible or list of them.
        @param values : Parameter values as float convertible or list of them.
        """
        debug('GCSCommands.CTO(lines=%r, params=%r, values=%r)', lines, params, values)
        checksize((True, True, True), lines, params, values)
        cmdstr = self.getcmdstr('CTO', lines, params, values)
        self.__msgs.send(cmdstr)

    def DRC(self, tables, sources, options):
        """Set data recorder configuration.
        @param tables : Record table IDs as integer convertible or list of them.
        @param sources : Record source IDs as string convertible or list of them.
        @param options : Parameter options as integer convertible or list of them.
        """
        debug('GCSCommands.DRC(tables=%r, sources=%r, options=%r)', tables, sources, options)
        checksize((True, True, True), tables, sources, options)
        cmdstr = self.getcmdstr('DRC', tables, sources, options)
        self.__msgs.send(cmdstr)

    def qDRC(self, tables=None):
        """Get the data recorder configuration for the queried record 'tables'.
        @param tables : Record table IDs as integer convertible or list of them.
        @return : Ordered dictionary of {table: (source, option)}, source is str, others int.
        """
        debug('GCSCommands.qDRC(tables=%r)', tables)
        cmdstr = self.getcmdstr('DRC?', tables)
        answer = self.__msgs.read(cmdstr)
        answerdict = getdict_oneitem(answer, tables, valueconv=(str, int), itemconv=int)
        debug('GCSCommands.qDRC = %r', answerdict)
        return answerdict

    def SGA(self, channels, values=None):
        """Set the gain value for the given analog input channel.
        @param channels: Channel or list of channels or dictionary {channel : value}.
        @param values : Parameter values as float convertible or list of them.
        """
        debug('GCSCommands.SGA(channels=%r, values=%r)', channels, values)
        channels, values = getitemsvaluestuple(channels, values)
        cmdstr = self.getcmdstr('SGA', channels, values)
        self.__msgs.send(cmdstr)

    def NAV(self, channels, values=None):
        """Set the number of readout values of the analog input that are averaged.
        @param channels: Channel or list of channels or dictionary {channel : value}.
        @param values : Parameter values as integer convertible or list of them.
        """
        debug('GCSCommands.NAV(channels=%r, values=%r)', channels, values)
        channels, values = getitemsvaluestuple(channels, values)
        cmdstr = self.getcmdstr('NAV', channels, values)
        self.__msgs.send(cmdstr)

    def DTC(self, tables):
        """Clear the given DDL tables.
        @param tables : Table IDs as integer convertible or list of them.
        """
        debug('GCSCommands.DTC(tables=%r)', tables)
        checksize((True,), tables)
        cmdstr = self.getcmdstr('DTC', tables)
        self.__msgs.send(cmdstr)

    def EGE(self, axes, values=None):
        """Enable electronic gearing for the specified 'axes'.
        @param axes: Axis or list of axes or dictionary {axis : value}.
        @param values : Bool convertible or list of them or None.
        """
        debug('GCSCommands.EGE(axes=%r, values=%r)', axes, values)
        axes, values = getitemsvaluestuple(axes, values)
        cmdstr = self.getcmdstr('EGE', axes, values)
        self.__msgs.send(cmdstr)

    def MRT(self, axes, values=None):
        """Execute a relative move in the tool coordinate system.
        Moves given axes relative to the current position and orientation
        of the tool coordinate system. Position and orientation of the tool
        coordinate system changes with each movement!
        Target position results from calculating the translation first and
        then the rotation. The order of rotation is U, V, W.
        @param axes: Axis or list of axes or dictionary {axis : value}.
        @param values : Float or list of floats or None.
        """
        debug('GCSCommands.MRT(axes=%r, values=%r)', axes, values)
        axes, values = getitemsvaluestuple(axes, values)
        cmdstr = self.getcmdstr('MRT', axes, values)
        self.__msgs.send(cmdstr)

    def MRW(self, axes, values=None):
        """Execute a relative move in the work coordinate system.
        Moves given axes relative to the current position and orientation of the work coordinate
        system. Position and orientation of the work coordinate system changes with each movement!
        Target position results from calculating the translation first and then the rotation. The
        order of rotation is U, V, W.
        @param axes: Axis or list of axes or dictionary {axis : value}.
        @param values : Float or list of floats or None.
        """
        debug('GCSCommands.MRW(axes=%r, values=%r)', axes, values)
        axes, values = getitemsvaluestuple(axes, values)
        cmdstr = self.getcmdstr('MRW', axes, values)
        self.__msgs.send(cmdstr)

    def MVE(self, axes, values=None):
        """Set new absolute target positions for given 'axes'.
        Axes will start moving to the new positions if ALL given
        targets are within the allowed range and ALL axes can move. If the
        affected axes are mounted in a way that they move perpendicular to
        each other, the combined motion of them will describe a linear path.
        This is achieved by appropriate calculation of accelerations, velocities
        and decelerations. The current settings for velocity, acceleration and
        deceleration define the maximum possible values, and the slowest axis
        determines the resulting velocities. All axes start moving
        simultaneously. This command can be interrupted by STP() and HLT().
        No other motion commands (e.g. MOV(), MVR()) are allowed during vector
        move. Servo must be enabled for all commanded axes prior to using this
        command. If servo is switched off or motion error occurs during motion,
        all axes are stopped.
        @param axes: Axis or list of axes or dictionary {axis : value}.
        @param values : Float or list of floats or None.
        """
        debug('GCSCommands.MVE(axes=%r, values=%r)', axes, values)
        axes, values = getitemsvaluestuple(axes, values)
        cmdstr = self.getcmdstr('MVE', axes, values)
        self.__msgs.send(cmdstr)

    def MVR(self, axes, values=None):
        """Move 'axes' relative to current target position.
        The new target position is calculated by adding the given
        position value to the last commanded target value. Axes will start
        moving to the new position if ALL given targets are within the allowed
        range and ALL axes can move. All axes start moving simultaneously. Servo
        must be enabled for all commanded axes prior to using this command.
        @param axes: Axis or list of axes or dictionary {axis : value}.
        @param values : Float or list of floats or None.
        """
        debug('GCSCommands.MVR(axes=%r, values=%r)', axes, values)
        axes, values = getitemsvaluestuple(axes, values)
        cmdstr = self.getcmdstr('MVR', axes, values)
        self.__msgs.send(cmdstr)

    def NLM(self, axes, values=None):
        """Set lower limits ("soft limit") for the positions of 'axes'.
        Depending on the controller, the soft limits are activated and
        deactivated with SSL().
        @param axes: Axis or list of axes or dictionary {axis : value}.
        @param values : Float or list of floats or None.
        """
        debug('GCSCommands.NLM(axes=%r, values=%r)', axes, values)
        axes, values = getitemsvaluestuple(axes, values)
        cmdstr = self.getcmdstr('NLM', axes, values)
        self.__msgs.send(cmdstr)

    def OMA(self, axes, values=None):
        """Command 'axes' to the given absolute position.
        Motion is realized in open-loop nanostepping mode. Servo must
        be disabled for the commanded axis prior to using this function
        (open-loop operation). With OMA() there is no position control (i.e.
        the target position is not maintained by any control loop).
        @param axes: Axis or list of axes or dictionary {axis : value}.
        @param values : Float or list of floats or None.
        """
        debug('GCSCommands.OMA(axes=%r, values=%r)', axes, values)
        axes, values = getitemsvaluestuple(axes, values)
        cmdstr = self.getcmdstr('OMA', axes, values)
        self.__msgs.send(cmdstr)

    def OMR(self, axes, values=None):
        """Command 'axes' relative to a position.
        Commands 'axes' to a position relative to the last commanded
        open-loop target position. The new open-loop target position is
        calculated by adding the given 'values' to the last commanded target
        value. Motion is realized in nanostepping mode. Servo must be disabled
        for the commanded axis prior to using this function (open-loop
        operation). With OMR there is no position control (i.e. the target
        position is not maintained by a control loop).
        @param axes: Axis or list of axes or dictionary {axis : value}.
        @param values : Float or list of floats or None.
        """
        debug('GCSCommands.OMR(axes=%r, values=%r)', axes, values)
        axes, values = getitemsvaluestuple(axes, values)
        cmdstr = self.getcmdstr('OMR', axes, values)
        self.__msgs.send(cmdstr)

    def PLM(self, axes, values=None):
        """Set upper limits ("soft limit") for the positions of 'axes'.
        Depending on the controller, the soft limits are activated
        and deactivated with SSL().
        @param axes: Axis or list of axes or dictionary {axis : value}.
        @param values : Float or list of floats or None.
        """
        debug('GCSCommands.PLM(axes=%r, values=%r)', axes, values)
        axes, values = getitemsvaluestuple(axes, values)
        cmdstr = self.getcmdstr('PLM', axes, values)
        self.__msgs.send(cmdstr)

    def RPA(self, items=None, params=None):
        """Reset the parameter 'params' of 'item'.
        The value of the EPROM is written into the RAM. RPA doesn't care about the command level.
        If the command is set at CCL0, CCL1 parameters are also reset. (Because powering the
        controller off and on also will reset them parameters.) If no parameters are given, all
        parameters for all items are reset.
        @param items: Axis/channel or list of them or dictionary {item : param} or None.
        @param params : Integer convertible or list of integer convertibles or None. Preceed
        hexadecimal strings with "0x" or "0X", e.g. "0xaffe".
        """
        debug('GCSCommands.RPA(items=%r, params=%r)', items, params)
        items, params = getitemsvaluestuple(items, params, required=False)
        cmdstr = self.getcmdstr('RPA', items, params)
        self.__msgs.send(cmdstr)

    def SMO(self, axes, values=None):
        """Set motor output. Value range depends on device. See controller manual.
        @param axes: Axis or list of axes or dictionary {axis : value}.
        @param values : Integer convertible or list of integer convertibles or None.
        """
        debug('GCSCommands.SMO(axes=%r, values=%r)', axes, values)
        axes, values = getitemsvaluestuple(axes, values)
        cmdstr = self.getcmdstr('SMO', axes, values)
        self.__msgs.send(cmdstr)

    def SVA(self, axes, values=None):
        """Set absolute open-loop control value to move 'axes'.
        Servo must be switched off (open-loop operation) when using this command.
        @param axes: Axis or list of axes or dictionary {axis : value}.
        @param values : Float or list of floats or None.
        """
        debug('GCSCommands.SVA(axes=%r, values=%r)', axes, values)
        axes, values = getitemsvaluestuple(axes, values)
        cmdstr = self.getcmdstr('SVA', axes, values)
        self.__msgs.send(cmdstr)

    def SVR(self, axes, values=None):
        """Set open-loop control value relative to the current value.
        The new open-loop control value is calculated by adding the given value to the last
        commanded open-loop control value. Servo must be switched off when using this command
        (open-loop operation).
        @param axes: Axis or list of axes or dictionary {axis : value}.
        @param values : Float or list of floats or None.
        """
        debug('GCSCommands.SVR(axes=%r, values=%r)', axes, values)
        axes, values = getitemsvaluestuple(axes, values)
        cmdstr = self.getcmdstr('SVR', axes, values)
        self.__msgs.send(cmdstr)

    def VMA(self, channels, values=None):
        """Set upper PZT voltage soft limit of given piezo channels as 'channels'.
        @param channels: Channel or list of channels or dictionary {channel : value}.
        @param values : Float or list of floats or None.
        """
        debug('GCSCommands.VMA(channels=%r, values=%r)', channels, values)
        channels, values = getitemsvaluestuple(channels, values)
        cmdstr = self.getcmdstr('VMA', channels, values)
        self.__msgs.send(cmdstr)

    def VMI(self, channels, values=None):
        """Set lower PZT voltage soft limit of given piezo channels as 'channels'.
        @param channels: Channel or list of channels or dictionary {channel : value}.
        @param values : Float or list of floats or None.
        """
        debug('GCSCommands.VMI(channels=%r, values=%r)', channels, values)
        channels, values = getitemsvaluestuple(channels, values)
        cmdstr = self.getcmdstr('VMI', channels, values)
        self.__msgs.send(cmdstr)

    def qATZ(self, axes=None):
        """Get if the AutoZero procedure called by ATZ() was successful.
        @param axes : String convertible or list of them or None.
        @return : Ordered dictionary of {axis: value}, values are bool.
        """
        debug('GCSCommands.qATZ(axes=%r)', axes)
        cmdstr = self.getcmdstr('ATZ?', axes)
        answer = self.__msgs.read(cmdstr)
        answerdict = getdict_oneitem(answer, axes, valueconv=(bool,))
        debug('GCSCommands.qATZ = %r', answerdict)
        return answerdict

    def qBRA(self, axes=None):
        """Get brake activation state of given 'axes'.
        @param axes : String convertible or list of them or None.
        @return : Ordered dictionary of {axis: value}, values are bool.
        """
        debug('GCSCommands.qBRA(axes=%r)', axes)
        cmdstr = self.getcmdstr('BRA?', axes)
        answer = self.__msgs.read(cmdstr)
        answerdict = getdict_oneitem(answer, axes, valueconv=(bool,))
        debug('GCSCommands.qBRA = %r', answerdict)
        return answerdict

    def qMVT(self, axes=None):
        """Get the current "move triggered" mode setting of the given 'axes'.
        @param axes : String convertible or list of them or None.
        @return : Ordered dictionary of {axis: value}, values are bool.
        """
        debug('GCSCommands.qMVT(axes=%r)', axes)
        cmdstr = self.getcmdstr('MVT?', axes)
        answer = self.__msgs.read(cmdstr)
        answerdict = getdict_oneitem(answer, axes, valueconv=(bool,))
        debug('GCSCommands.qMVT = %r', answerdict)
        return answerdict

    def qSTE(self, axes=None):
        """Get the last sent amplitude for the step response measurement for given 'axes'.
        @param axes : String convertible or list of them or None.
        @return : Ordered dictionary of {axis: value}, values are float.
        """
        debug('GCSCommands.qSTE(axes=%r)', axes)
        cmdstr = self.getcmdstr('STE?', axes)
        answer = self.__msgs.read(cmdstr)
        answerdict = getdict_oneitem(answer, axes, valueconv=(float,))
        debug('GCSCommands.qSTE = %r', answerdict)
        return answerdict

    def qIMP(self, axes=None):
        """Get the last sent impulse parameters for given 'axes'.
        @param axes : String convertible or list of them or None.
        @return : Ordered dictionary of {axis: value}, values are float.
        """
        debug('GCSCommands.qIMP(axes=%r)', axes)
        cmdstr = self.getcmdstr('IMP?', axes)
        answer = self.__msgs.read(cmdstr)
        answerdict = getdict_oneitem(answer, axes, valueconv=(float,))
        debug('GCSCommands.qIMP = %r', answerdict)
        return answerdict

    def qCMO(self, axes=None):
        """Get the closed-loop control mode which is currently selected for the 'axes'.
        @param axes : String convertible or list of them or None.
        @return : Ordered dictionary of {axis: value}, values are int.
        """
        debug('GCSCommands.qCMO(axes=%r)', axes)
        cmdstr = self.getcmdstr('CMO?', axes)
        answer = self.__msgs.read(cmdstr)
        answerdict = getdict_oneitem(answer, axes, valueconv=(int,))
        debug('GCSCommands.qCMO = %r', answerdict)
        return answerdict

    def qOMA(self, axes=None):
        """Get the last commanded open-loop targets of given 'axes'.
        @param axes : String convertible or list of them or None.
        @return : Ordered dictionary of {axis: value}, values are float.
        """
        debug('GCSCommands.qOMA(axes=%r)', axes)
        cmdstr = self.getcmdstr('OMA?', axes)
        answer = self.__msgs.read(cmdstr)
        answerdict = getdict_oneitem(answer, axes, valueconv=(float,), itemconv=int)
        debug('GCSCommands.qOMA = %r', answerdict)
        return answerdict

    def qCTV(self, axes=None):
        """Get closed-loop target. Returns the currently valid value.
        @param axes : String convertible or list of them or None.
        @return : Ordered dictionary of {axis: value}, values are float.
        """
        debug('GCSCommands.qCTV(axes=%r)', axes)
        cmdstr = self.getcmdstr('CTV?', axes)
        answer = self.__msgs.read(cmdstr)
        answerdict = getdict_oneitem(answer, axes, valueconv=(float,))
        debug('GCSCommands.qCTV = %r', answerdict)
        return answerdict

    def qSMO(self, axes=None):
        """Get the set values for the motor output.
        @param axes : String convertible or list of them or None.
        @return : Ordered dictionary of {axis: value}, values are int.
        """
        debug('GCSCommands.qSMO(axes=%r)', axes)
        cmdstr = self.getcmdstr('SMO?', axes)
        answer = self.__msgs.read(cmdstr)
        answerdict = getdict_oneitem(answer, axes, valueconv=(int,))
        debug('GCSCommands.qSMO = %r', answerdict)
        return answerdict

    def qSRA(self, axes=None):
        """Get the electronic gear ratio for 'axes'.
        @param axes : String convertible or list of them or None.
        @return : Ordered dictionary of {axis: value}, values are float.
        """
        debug('GCSCommands.qSRA(axes=%r)', axes)
        cmdstr = self.getcmdstr('SRA?', axes)
        answer = self.__msgs.read(cmdstr)
        answerdict = getdict_oneitem(answer, axes, valueconv=(float,))
        debug('GCSCommands.qSRA = %r', answerdict)
        return answerdict

    def AVG(self, value):
        """Set the number of samples used to calculate an average to 'value'.
        We recommend a value between 1 and 100.
        @param value : Number of samples as integer.
        """
        debug('GCSCommands.AVG(value=%r)', value)
        checksize((1,), value)
        cmdstr = self.getcmdstr('AVG', value)
        self.__msgs.send(cmdstr)

    def JLT(self, joystick, joyaxis, offset, values):
        """Set values in joystick lookup table to 'values'.
        @param joystick : Joystick device ID as integer.
        @param joyaxis : Joystick axis ID as integer.
        @param offset : Start point in the lookup table as integer, starts with index 1.
        @param values : Value or list of values as float, range -1.0..1.0.
        """
        debug('GCSCommands.JLT(joystick=%s, joyaxis=%s, offset=%s, value=%r)', joystick, joyaxis, offset, values)
        checksize((1, 1, 1, True), joystick, joyaxis, offset, values)
        cmdstr = self.getcmdstr('JLT', joystick, joyaxis, offset, values)
        self.__msgs.send(cmdstr)

    def CSV(self, value):
        """Set the GCS syntax version to 'value'.
        @param value : GCS version as integer or float, e.g. 1 or 2.
        """
        debug('GCSCommands.CSV(value=%r)', value)
        checksize((1,), value)
        cmdstr = self.getcmdstr('CSV', value)
        self.__msgs.send(cmdstr)

    def BDR(self, value):
        """Set the RS-232 communications baud rate of the master. This will only change the setting
        in the RAM. To store it in the EEPROM call WPA() afterwards. After the next start of the
        controller the new setting will be used. If you want to change it immediately, call RST()
        after WPA().
        @param value : Target baud rate as integer.
        """
        debug('GCSCommands.BDR(value=%r)', value)
        checksize((1,), value)
        cmdstr = self.getcmdstr('BDR', value)
        self.__msgs.send(cmdstr)

    def CCL(self, level, password=''):
        """Set the command level of the controller.
        This determines the availability of commands and the write access to the system parameters.
        @param level: Command level to set as integer convertible.
        @param password : Password to access 'level' as string, defaults to no password.
        """
        debug('GCSCommands.CCL(level=%r, password=%r)', level, password)
        checksize((1,), level)
        if password:
            checksize((1,), password)
        cmdstr = self.getcmdstr('CCL', level, password)
        self.__msgs.send(cmdstr)
        del self.funcs

    def CLR(self, axes=None):
        """Clear the status of 'axes'.
        The following actions are done by CLR(): Switches the servo on.
        Resets error to 0. If the stage has tripped a limit switch, CLR() will
        move it away from the limit switch until the limit condition is no
        longer given, and the target position is set to the current position
        afterwards.
        @param axes : String convertible or list of them or None.
        """
        debug('GCSCommands.CLR(axes=%r)', axes)
        axes = getitemslist(axes)
        cmdstr = self.getcmdstr('CLR', axes)
        self.__msgs.send(cmdstr)

    def CPY(self, name, cmd=None):
        """Copy a command response into a variable. Local variables can be set
        using CPY in macros only. The variable is present in RAM only.
        @param name: Single name as string or dictionary {name : cmd}.
        @param cmd : String of command or None.
        """
        debug('GCSCommands.CPY(name=%r, cmd=%r)', name, cmd)
        name, cmd = getitemsvaluestuple(name, cmd)
        cmdstr = self.getcmdstr('CPY', name, cmd)
        self.__msgs.send(cmdstr)

    def MAC_DEF(self, item=''):
        """Set macro with name 'item' as start-up macro.
        This macro will be automatically executed with the next
        power-on or reboot of the controller. If 'item' is omitted, the current
        start-up macro selection is canceled. To find out what macros are
        available call qMAC().
        @param item : Item name as string or empty or None to delete current setting.
        """
        debug('GCSCommands.MAC_DEF(item=%r)', item)
        if item:
            checksize((1,), item)
        else:
            item = ''
        cmdstr = self.getcmdstr('MAC DEF', item)
        self.__msgs.send(cmdstr)

    def REF(self, axes=None):
        """Reference 'axes'.
        Error check will be disabled temporarily for GCS1 devices.
        @param axes : axes names as string.
        """
        debug('GCSCommands.REF(axes=%r)', axes)
        errcheck = self.__msgs.errcheck
        if not self.isgcs2:
            self.__msgs.errcheck = False
        cmdstr = self.getcmdstr('REF', axes)
        self.__msgs.send(cmdstr)
        if not self.isgcs2:
            self.__msgs.errcheck = errcheck

    def RBT(self):
        """Reboot controller, error check will be disabled temporarily."""
        debug('GCSCommands.RBT()')
        errcheck = self.__msgs.errcheck
        self.__msgs.errcheck = False
        cmdstr = self.getcmdstr('RBT', )
        self.__msgs.send(cmdstr)
        self.__msgs.errcheck = errcheck

    def SAI(self, oldaxes, newaxes=None):
        """Rename axes.
        The characters in 'newaxes' must not be in use for any other existing axes and must each
        be one of the valid identifiers. All characters in 'newaxes' will be converted to upper
        case letters.
        @param oldaxes: Name or list of them or dictionary {oldname : newname}.
        @param newaxes : Name or list of names or None.
        """
        debug('GCSCommands.SAI(oldaxes=%r, newaxes=%r)', oldaxes, newaxes)
        oldaxes, newaxes = getitemsvaluestuple(oldaxes, newaxes)
        cmdstr = self.getcmdstr('SAI', oldaxes, newaxes)
        self.__msgs.send(cmdstr)
        self.__axes = []

    def qAVG(self):
        """Get the number of samples used to calculate an average.
        @return : Number of samples as integer.
        """
        debug('GCSCommands.qAVG()')
        answer = self.__msgs.read('AVG?')
        value = int(answer.strip())
        debug('GCSCommands.qAVG = %r', value)
        return value

    def qBDR(self):
        """Get the current RAM baud rate setting of the master.
        This is the value that will be saved to ROM by WPA() and may
        differ from both the power-up and/or the current operating value.
        @return : Answer as integer.
        """
        debug('GCSCommands.qBDR()')
        answer = self.__msgs.read('BDR?')
        value = int(answer.strip())
        debug('GCSCommands.qBDR = %r', value)
        return value

    def qRTR(self):
        """Get the record table rate, i.e. the number of servo-loop cycles to be used in data.
        recording operations .
        @return : Answer as integer.
        """
        debug('GCSCommands.qRTR()')
        answer = self.__msgs.read('RTR?')
        value = int(answer.strip())
        debug('GCSCommands.qRTR = %r', value)
        return value

    def qSCT(self):
        """Get the cycle time of the trajectory generator in milliseconds.
        @return : Answer as float.
        """
        debug('GCSCommands.qSCT()')
        answer = self.__msgs.read('SCT?')
        value = float(answer.strip().split('=')[1])
        debug('GCSCommands.qSCT = %r', value)
        return value

    def qSCH(self):
        """Get the axis identifier of the master.
        @return : Answer as string with trailing linefeed.
        """
        debug('GCSCommands.qSCH()')
        answer = self.__msgs.read('SCH?')
        debug('GCSCommands.qSCH = %r', answer)
        return answer

    def qVST(self):
        """Get the names of all stages which can be connected to the controller.
        @return : Answer as list of strings without trailing linefeeds.
        """
        debug('GCSCommands.qVST()')
        answer = self.__msgs.read('VST?')
        answer = splitparams(answer, self.isgcs2)
        debug('GCSCommands.qVST = %r', answer)
        return answer

    def qSSN(self):
        """Get the serial number of the controller.
        @return : Answer as string with trailing linefeed.
        """
        debug('GCSCommands.qSSN()')
        answer = self.__msgs.read('SSN?')
        debug('GCSCommands.qSSN = %r', answer)
        return answer

    def qHIS(self, devices=None, items=None, properties=None):
        """Get human interface device (HID) items connected to the controller.
        @param devices : ID of human interface device as int or list of ints.
        @param items : ID of one item of the HID as int or list of ints.
        @param properties : ID of property to query as int or list of ints.
        @return : Ordered dictionary of {(device, item, property): value}, devices/items/properties are int,
        values are string.
        """
        debug('GCSCommands.qHIS(devices=%s, items=%s, properties=%s)', devices, items, properties)
        if devices or items or properties:
            checksize((), devices, items, properties)
        cmdstr = self.getcmdstr('HIS?', devices, items, properties)
        answerdict = OrderedDict()
        answer = self.__msgs.read(cmdstr)
        if answer.strip():
            for line in answer.splitlines():
                value = line.split('=')[1].strip()
                device, item, prop = line.split('=')[0].split()
                answerdict[(int(device), int(item), int(prop))] = value
        debug('GCSCommands.qHIS = %r', answerdict)
        return answerdict

    def qHDI(self):
        """Get help on diagnosis information, received with DIA().
        @return : Answer as string with trailing linefeed.
        """
        debug('GCSCommands.qHDI()')
        answer = self.__msgs.read('HDI?')
        debug('GCSCommands.qHDI = %r', answer)
        return answer

    def qHPV(self):
        """Get help string about possible parameter values.
        @return : Answer as string with trailing linefeed.
        """
        debug('GCSCommands.qHPV()')
        answer = self.__msgs.read('HPV?')
        debug('GCSCommands.qHPV = %r', answer)
        return answer

    def qCCL(self):
        """Get the current command level.
        @return : Current command level as integer.
        """
        debug('GCSCommands.qCCL()')
        answer = self.__msgs.read('CCL?')
        value = int(answer.strip())
        debug('GCSCommands.qCCL = %r', value)
        return value

    def qCSV(self):
        """Get the current GCS syntax version.
        @return : GCS version as float.
        """
        debug('GCSCommands.qCSV()')
        answer = self.__msgs.read('CSV?')
        value = float(answer.strip())
        debug('GCSCommands.qCSV = %r', value)
        return value

    def qERR(self):
        """Get current controller error.
        @return : Current error code as integer.
        """
        debug('GCSCommands.qERR()')
        errcheck = self.__msgs.errcheck
        self.__msgs.errcheck = False
        answer = self.__msgs.read('ERR?')
        self.__msgs.errcheck = errcheck
        try:
            value = int(answer.strip())
        except ValueError:
            raise pipython.GCSError(pipython.gcserror.E_1004_PI_UNEXPECTED_RESPONSE, answer)
        debug('GCSCommands.qERR = %r', value)
        return value

    def qSWT(self, channel, index):
        """Get a single value from the E-816 wave table data.
        @param channel : Controller index as string.
        @param index : Index for wave table entry as integer, starts with 0.
        @return : Wave table point value as float.
        """
        debug('GCSCommands.qSWT(channel=%r, index=%r)', channel, index)
        checksize((1, 1), channel, index)
        cmdstr = self.getcmdstr('SWT?', channel, index)
        answer = self.__msgs.read(cmdstr)
        value = float(answer.strip())
        debug('GCSCommands.qSWT = %r', value)
        return value

    def MAC_qFREE(self):
        """Get free memory memory for macros in number of characters.
        @return : Free memory as integer.
        """
        debug('GCSCommands.MAC_qFREE()')
        answer = self.__msgs.read('MAC FREE?')
        value = int(answer.strip())
        debug('GCSCommands.MAC_qFREE = %r', value)
        return value

    def qFSS(self):
        """Get result of last scan.
        @return : 1 if scan has finished successfully, i.e. threshold or a maximum was found.
        """
        debug('GCSCommands.qFSS()')
        answer = self.__msgs.read('FSS?')
        value = int(answer.strip())
        debug('GCSCommands.qFSS = %r', value)
        return value

    def qGFL(self):
        """Get number of recorded frequency measurements.
        @return : Number of recorded frequency measurements as integer.
        """
        debug('GCSCommands.qGFL()')
        answer = self.__msgs.read('GFL?')
        value = int(answer.strip())
        debug('GCSCommands.qGFL = %r', value)
        return value

    def qTAC(self):
        """Get number of installed analog channels.
        @return : Current value as integer.
        """
        debug('GCSCommands.qTAC()')
        answer = self.__msgs.read('TAC?')
        value = int(answer.strip())
        debug('GCSCommands.qTAC = %r', value)
        return value

    def qTIM(self):
        """Get the time in milliseconds since start-up or last TIM().
        @return Current value as float.
        """
        debug('GCSCommands.qTIM()')
        answer = self.__msgs.read('TIM?')
        value = float(answer.strip())
        debug('GCSCommands.qTIM = %r', value)
        return value

    def qTIO(self):
        """Get the number of digital inputs and outputs available on the controller.
        @return Dictionary {'I': value, 'O': value}, values as integers.
        """
        debug('GCSCommands.qTIO()')
        answer = self.__msgs.read('TIO?')
        # noinspection PyTypeChecker
        answerdict = getdict_oneitem(answer, items=None, valueconv=(int,))
        debug('GCSCommands.qTIO = %r', answerdict)
        return answerdict

    def qWFR(self):
        """Get parameters of the last WFR command.
        @return Dictionary {item: value}, item is string, values are converted
        to an according type (string, int, float).
        """
        debug('GCSCommands.qWFR()')
        answer = self.__msgs.read('WFR?')
        # noinspection PyTypeChecker
        answerdict = getdict_oneitem(answer, items=None, valueconv=(True,))
        debug('GCSCommands.qWFR = %r', answerdict)
        return answerdict

    def qTLT(self):
        """Get the number of DDL tables.
        @return Current value as integer.
        """
        debug('GCSCommands.qTLT()')
        answer = self.__msgs.read('TLT?')
        value = int(answer.strip())
        debug('GCSCommands.qTLT = %r', value)
        return value

    def qTNR(self):
        """Get the number of data recorder tables currently available on the controller.
        @return Current value as integer.
        """
        debug('GCSCommands.qTNR()')
        answer = self.__msgs.read('TNR?')
        value = int(answer.strip())
        debug('GCSCommands.qTNR = %r', value)
        return value

    def qTNJ(self):
        """Get the number of joysticks.
        @return Current value as integer.
        """
        debug('GCSCommands.qTNJ()')
        answer = self.__msgs.read('TNJ?')
        value = int(answer.strip())
        debug('GCSCommands.qTNJ = %r', value)
        return value

    def qTPC(self):
        """Get the number of piezo channels.
        @return Current value as integer.
        """
        debug('GCSCommands.qTPC()')
        answer = self.__msgs.read('TPC?')
        value = int(answer.strip())
        debug('GCSCommands.qTPC = %r', value)
        return value

    def qTSC(self):
        """Get the number of sensor channels.
        @return Current value as integer.
        """
        debug('GCSCommands.qTSC()')
        answer = self.__msgs.read('TSC?')
        value = int(answer.strip())
        debug('GCSCommands.qTSC = %r', value)
        return value

    def qTWG(self):
        """Get the number of wave generators.
        @return Current value as integer.
        """
        debug('GCSCommands.qTWG()')
        answer = self.__msgs.read('TWG?')
        value = int(answer.strip())
        debug('GCSCommands.qTWG = %r', value)
        return value

    def qVLS(self):
        """Get the current maximum system velocity.
        @return Current value as float.
        """
        debug('GCSCommands.qVLS()')
        answer = self.__msgs.read('VLS?')
        value = float(answer.strip())
        debug('GCSCommands.qVLS = %r', value)
        return value

    def qIDN(self):
        """Get controller identification.
        @return : Controller ID as string with trailing linefeed.
        """
        debug('GCSCommands.qIDN()')
        answer = self.__msgs.read('*IDN?')
        debug('GCSCommands.qIDN = %r', answer)
        return answer

    def qVER(self):
        """Get version information about firmware and modules.
        @return : Version information as string with trailing linefeeds.
        """
        debug('GCSCommands.qVER()')
        answer = self.__msgs.read('VER?')
        answer = answer[:-1] + ' \nPIPython: %s\n' % pipython.__version__
        debug('GCSCommands.qVER = %r', answer)
        return answer

    def qTVI(self):
        """Get valid axis identifiers. Should be called before axes are renamed with SAI().
        @return : Valid axis identifiers as string with trailing linefeed.
        """
        debug('GCSCommands.qTVI()')
        answer = self.__msgs.read('TVI?')
        debug('GCSCommands.qTVI = %r', answer)
        return answer

    def qVAR(self, varname=None):
        """Get variable value.
        If qVAR is combined with CPY(), JRC(), MEX() or WAC(), the
        response to qVAR() has to be a single value and not more.
        @param varname : Name of the variable as string or list of them.
        @return Dictionary {name: value}, value of type string.
        """
        debug('GCSCommands.qVAR(varname=%r)', varname)
        cmdstr = self.getcmdstr('VAR?', varname)
        answer = self.__msgs.read(cmdstr)
        answerdict = getdict_oneitem(answer, varname, valueconv=(str,))
        debug('GCSCommands.qVAR = %r', answerdict)
        return answerdict

    def BRA(self, axes, values=None):
        """Set the brake state for 'axes' to on (True) or off (False).
        @param axes: Axis or list of axes or dictionary {axis : value}.
        @param values : Bool or list of bools or None.
        """
        debug('GCSCommands.BRA(axes=%r, values=%r)', axes, values)
        axes, values = getitemsvaluestuple(axes, values)
        cmdstr = self.getcmdstr('BRA', axes, values)
        self.__msgs.send(cmdstr)

    def qCST(self, axes=None):
        """Get the type names of the stages associated with 'axes'. If axes is None the connected
        axes are used. Call qCST(qSAI_ALL()) to query all axes, i.e. including NOSTAGE axes.
        @param axes : Can be string convertible or list of them or None.
        @return : Ordered dictionary {axis: stagename}
        """
        debug('GCSCommands.qCST(varname=%r)', axes)
        cmdstr = self.getcmdstr('CST?', axes)
        answer = self.__msgs.read(cmdstr)
        answerdict = getdict_oneitem(answer, axes, valueconv=(str,))
        debug('GCSCommands.qCST = %r', answerdict)
        return answerdict

    def qPUN(self, axes=None):
        """Get the position units of 'axes'.
        @param axes : Can be string convertible or list of them or None.
        @return : Ordered dictionary {axis: unit}, unit as string.
        """
        debug('GCSCommands.qPUN(varname=%r)', axes)
        cmdstr = self.getcmdstr('PUN?', axes)
        answer = self.__msgs.read(cmdstr)
        answerdict = getdict_oneitem(answer, axes, valueconv=(str,))
        debug('GCSCommands.qPUN = %r', answerdict)
        return answerdict

    def qECO(self, sendstring):
        """Get 'sendstring' as upper case string, can be used to test communication.
        @param sendstring : Any string to be returned by controller.
        @return : Received string with trailing linefeed.
        """
        debug('GCSCommands.qIDN()')
        checksize((1,), sendstring)
        cmdstr = self.getcmdstr('ECO?', sendstring)
        answer = self.__msgs.read(cmdstr)
        debug('GCSCommands.qECO = %r', answer)
        return answer

    def qKEN(self, csnames=None):
        """Get types of enabled coordinate systems.
        @param csnames : Name or list of the coordinate system(s) as string.
        @return:  Ordered dictionary {csname: type}.
        """
        debug('GCSCommands.qKEN(csnames=%r)', csnames)
        cmdstr = self.getcmdstr('KEN?', csnames)
        answer = self.__msgs.read(cmdstr)
        answerdict = getdict_oneitem(answer, csnames, valueconv=(str,))
        debug('GCSCommands.qKEN = %r', answerdict)
        return answerdict

    def qKLN(self, csnames=None):
        """Get chains of coordinate systems.
        @param csnames : Name or list of the node(s) as string.
        @return : Ordered dictionary {name: chain}.
        """
        debug('GCSCommands.qKLN(csnames=%r)', csnames)
        cmdstr = self.getcmdstr('KLN?', csnames)
        answer = self.__msgs.read(cmdstr)
        answerdict = getdict_oneitem(answer, csnames, valueconv=(str,))
        debug('GCSCommands.qKLN = %r', answerdict)
        return answerdict

    def qKET(self, cstypes=None):
        """Get enabled coordinate systems of cstypes.
        @param cstypes: Coordinate system type or list as string.
        @return Ordered dictionary {type: name}.
        """
        debug('GCSCommands.qKET(cstypes=%r)', cstypes)
        cmdstr = self.getcmdstr('KET?', cstypes)
        answer = self.__msgs.read(cmdstr)
        answerdict = getdict_oneitem(answer, cstypes, valueconv=(str,))
        debug('GCSCommands.qKET = %r', answerdict)
        return answerdict

    def qDCO(self, axes=None):
        """Get drift compensation mode of 'axes'.
        @param axes : String convertible or list of them or None.
        @return : Ordered dictionary of {axis: value}, values are bool.
        """
        debug('GCSCommands.qDCO(axes=%r)', axes)
        cmdstr = self.getcmdstr('DCO?', axes)
        answer = self.__msgs.read(cmdstr)
        answerdict = getdict_oneitem(answer, axes, valueconv=(bool,))
        debug('GCSCommands.qDCO = %r', answerdict)
        return answerdict

    def qEAX(self, axes=None):
        """Get enabled states of 'axes'.
        @param axes : String convertible or list of them or None.
        @return : Ordered dictionary of {axis: value}, values are bool.
        """
        debug('GCSCommands.qEAX(axes=%r)', axes)
        cmdstr = self.getcmdstr('EAX?', axes)
        answer = self.__msgs.read(cmdstr)
        answerdict = getdict_oneitem(answer, axes, valueconv=(bool,))
        debug('GCSCommands.qEAX = %r', answerdict)
        return answerdict

    def qLIM(self, axes=None):
        """Ask if given axis has limit switches.
        True if stage has limit sensors and controller does support reading limit sensor signals.
        @param axes : String convertible or list of them or None.
        @return : Ordered dictionary of {axis: value}, values are bool.
        """
        debug('GCSCommands.qLIM(axes=%r)', axes)
        cmdstr = self.getcmdstr('LIM?', axes)
        answer = self.__msgs.read(cmdstr)
        answerdict = getdict_oneitem(answer, axes, valueconv=(bool,))
        debug('GCSCommands.qLIM = %r', answerdict)
        return answerdict

    def ONL(self, channels, values=None):
        """Set control mode for given piezo 'channels' (ONLINE or OFFLINE mode).
        @param channels: Lines as int or string or list of them or dictionary {channel : value}.
        @param values : Bool convertible or list of them or None.
        """
        debug('GCSCommands.ONL(channels=%r, values=%r)', channels, values)
        channels, values = getitemsvaluestuple(channels, values)
        cmdstr = self.getcmdstr('ONL', channels, values)
        self.__msgs.send(cmdstr)

    def WCL(self, tables):
        """Clear the content of the given wave 'tables'.
        @param tables : Wave table ID as integer convertible or list of them.
        """
        debug('GCSCommands.WCL(tables=%r)', tables)
        checksize((True,), tables)
        cmdstr = self.getcmdstr('WCL', tables)
        self.__msgs.send(cmdstr)

    def qDRL(self, tables=None):
        """Get the number of points comprised by the last recording.
        I.e. the number of values that have been recorded since data
        recording was last triggered. This way it is possible to find out if
        recording has been finished (all desired points are in the record table)
        or to find out how many points can be currently read from the record
        table. Depending on the controller, reading more points than the number
        returned by qDRL() can also read old record table content.
        @param tables : Integer convertible or list of them or None.
        @return : Ordered dictionary of {channel: value}, values are int.
        """
        debug('GCSCommands.qDRL(tables=%r)', tables)
        cmdstr = self.getcmdstr('DRL?', tables)
        answer = self.__msgs.read(cmdstr)
        answerdict = getdict_oneitem(answer, tables, valueconv=(int,), itemconv=int)
        debug('GCSCommands.qDRL = %r', answerdict)
        return answerdict

    def qWGC(self, wavegens=None):
        """Get the number of wave generator output cycles.
        @param wavegens : Integer convertible or list of them or None.
        @return : Ordered dictionary of {wavegen: value}, values are int.
        """
        debug('GCSCommands.qWGC(wavegens=%r)', wavegens)
        cmdstr = self.getcmdstr('WGC?', wavegens)
        answer = self.__msgs.read(cmdstr)
        answerdict = getdict_oneitem(answer, wavegens, valueconv=(int,), itemconv=int)
        debug('GCSCommands.qWGC = %r', answerdict)
        return answerdict

    def qWGO(self, wavegens=None):
        """Get the start/stop mode of the given wave generator.
        @param wavegens : Integer convertible or list of them or None.
        @return : Ordered dictionary of {wavegen: value}, values are int.
        """
        debug('GCSCommands.qWGO(wavegens=%r)', wavegens)
        cmdstr = self.getcmdstr('WGO?', wavegens)
        answer = self.__msgs.read(cmdstr)
        answerdict = getdict_oneitem(answer, wavegens, valueconv=(int,), itemconv=int)
        debug('GCSCommands.qWGO = %r', answerdict)
        return answerdict

    def qWMS(self, tables=None):
        """Get the maximum size of wave storage.
        @param tables : Integer convertible or list of them or None.
        @return : Ordered dictionary of {wavetable: value}, values are int.
        """
        debug('GCSCommands.qWMS(tables=%r)', tables)
        cmdstr = self.getcmdstr('WMS?', tables)
        answer = self.__msgs.read(cmdstr)
        answerdict = getdict_oneitem(answer, tables, valueconv=(int,), itemconv=int)
        debug('GCSCommands.qWMS = %r', answerdict)
        return answerdict

    def qWTR(self, wavegens=None):
        """Get the current wave generator table rate and interpolation type.
        The table rate is the number of servo-loop cycles used for wave generator output.
        @param wavegens : Integer convertible or list of them or None.
        @return : Ordered dictionary of {wavegen: (rate, type)}, values are int.
        """
        debug('GCSCommands.qWTR(wavegens=%r)', wavegens)
        cmdstr = self.getcmdstr('WTR?', wavegens)
        answer = self.__msgs.read(cmdstr)
        answerdict = getdict_oneitem(answer, wavegens, valueconv=(int,), itemconv=int)
        debug('GCSCommands.qWTR = %r', answerdict)
        return answerdict

    def qCTI(self, lines=None, params=None):
        """Get the trigger input configuration for given trigger input 'lines'.
        @param lines : Integer convertible or list of them or None.
        @param params : Integer that describes a trigger function, see manual.
        @return : Ordered dictionary of {line: {param: value}}, lines/params are int,
        values are string.
        """
        debug('GCSCommands.qCTI(lines=%r, params=%r)', lines, params)
        if params:
            checksize((), lines, params)
        cmdstr = self.getcmdstr('CTI?', lines, params)
        answer = self.__msgs.read(cmdstr)
        answerdict = getdict_twoitems(answer, lines, params, itemconv=[int, int], valueconv=(str,))
        debug('GCSCommands.qCTI = %r', answerdict)
        return answerdict

    def qCTO(self, lines=None, params=None):
        """Get the trigger output configuration for given trigger input 'lines'.
        @param lines : Integer convertible or list of them or None.
        @param params : Integer that describes a trigger function, see manual.
        @return : Ordered dictionary of {line: {param: value}}, lines/params are int,
        values are float.
        """
        debug('GCSCommands.qCTO(lines=%r, params=%r)', lines, params)
        if params:
            checksize((), lines, params)
        cmdstr = self.getcmdstr('CTO?', lines, params)
        answer = self.__msgs.read(cmdstr)
        answerdict = getdict_twoitems(answer, lines, params, itemconv=[int, int], valueconv=(str,))
        debug('GCSCommands.qCTO = %r', answerdict)
        return answerdict

    def qDIA(self, items=None):
        """Get diagnosis information.
        @param items : Item ID as integer convertible or list of them or None.
        @return : Ordered dictionary of {item: value}, items are int, values are string.
        """
        debug('GCSCommands.qDIA(items=%r)', items)
        cmdstr = self.getcmdstr('DIA?', items)
        answer = self.__msgs.read(cmdstr)
        answerdict = getdict_oneitem(answer, items, valueconv=(str,))
        debug('GCSCommands.qDIA = %r', answerdict)
        return answerdict

    def qDRT(self, tables=None):
        """Get the current trigger source setting for the given data recorder 'tables'.
        @param tables : Record table ID as integer convertible or list of them or None.
        @return : Ordered dictionary of {rectable: (source, value)}, tables/sources are int,
        values are string.
        """
        debug('GCSCommands.qDRT(tables=%r)', tables)
        cmdstr = self.getcmdstr('DRT?', tables)
        answer = self.__msgs.read(cmdstr)
        answerdict = getdict_oneitem(answer, tables, valueconv=(int, str), itemconv=int)
        debug('GCSCommands.qDRT = %r', answerdict)
        return answerdict

    def qHDT(self, devices=None, axes=None):
        """Get the assigned lookup table for the given axes of the given HID device.
        @param devices: Item ID as integer or list of them.
        @param axes : Axis ID as integer or list of them.
        @return : Ordered dictionary of {device: {axis: table}, all are integers.
        """
        debug('GCSCommands.qHDT(devices=%r, axes=%r)', devices, axes)
        if axes:
            checksize((), devices, axes)
        cmdstr = self.getcmdstr('HDT?', devices, axes)
        answer = self.__msgs.read(cmdstr)
        answerdict = getdict_twoitems(answer, devices, axes, itemconv=[int, int], valueconv=(int,))
        debug('GCSCommands.qHDT = %r', answerdict)
        return answerdict

    def qFSN(self, axes=None):
        """Get result of FSN().
        Report the highest value of of the analog input during the last scan and the position where
        it was detected.
        @param axes : String convertible or list of them or None.
        @return : Ordered dictionary of {axis: (position, maximum)}, positions/maximums are float.
        """
        debug('GCSCommands.qFSN(axes=%r)', axes)
        cmdstr = self.getcmdstr('FSN?', axes)
        answer = self.__msgs.read(cmdstr)
        answerdict = getdict_oneitem(answer, axes, valueconv=(float, float))
        debug('GCSCommands.qFSN = %r', answerdict)
        return answerdict

    def qFED(self, axes=None):
        """Get the parameters of the last find edge motion made with FED().
        @param axes : String convertible or list of them or None.
        @return : Ordered dictionary of {rectable: (source, value)}, sources are int,
        values are string.
        """
        debug('GCSCommands.qFED(axes=%r)', axes)
        cmdstr = self.getcmdstr('FED?', axes)
        answer = self.__msgs.read(cmdstr)
        answerdict = getdict_oneitem(answer, axes, valueconv=(int, str), itemconv=int)
        debug('GCSCommands.qFED = %r', answerdict)
        return answerdict

    def qHIA(self, axes='', functions=None):
        """Get the current control configuration for the given motion parameter of the given
        axis of the controller, i.e. the currently assigned axis of an HID device.
        @param axes: Axis as string convertible or list of them.
        @param functions : Integer convertible or list of them.
        @return : Ordered dictionary of {axis: {function: (device, axis)}}, axis is string, others
        are integers.
        """
        debug('GCSCommands.qHIA(axes=%r, functions=%r)', axes, functions)
        axes = getitemslist(axes)
        functions = getitemslist(functions, int)
        checksize((), axes, functions)
        cmdstr = self.getcmdstr('HIA?', axes, functions)
        answer = self.__msgs.read(cmdstr)
        answerdict = getdict_twoitems(answer, axes, functions, itemconv=[str, int],
                                      valueconv=(int, int))
        debug('GCSCommands.qHIA = %r', answerdict)
        return answerdict

    def qTWE(self, tables=None):
        """Get the trigger definition set with TWE().
        @param tables : Table ID as integer convertible or list of them.
        @return : Ordered dictionary of {table: (start, end)}, all are integers.
        """
        debug('GCSCommands.qTWE(tables=%r)', tables)
        cmdstr = self.getcmdstr('TWE?', tables)
        answer = self.__msgs.read(cmdstr)
        answerdict = getdict_oneitem(answer, tables, itemconv=int, valueconv=(int, int))
        debug('GCSCommands.qTWE = %r', answerdict)
        return answerdict

    def qHIB(self, devices=None, buttons=None):
        """Get the current state of the given button of the given HID device.
        @param devices: Device ID or list of devices.
        @param buttons : Integer convertible or list of them.
        @return : Ordered dictionary of {device: {button: value}}, all are integers.
        """
        debug('GCSCommands.qHIB(devices=%r, buttons=%r)', devices, buttons)
        if buttons:
            checksize((), devices, buttons)
        cmdstr = self.getcmdstr('HIB?', devices, buttons)
        answer = self.__msgs.read(cmdstr)
        answerdict = getdict_twoitems(answer, devices, buttons, itemconv=[int, int],
                                      valueconv=(int,))
        debug('GCSCommands.qHIB = %r', answerdict)
        return answerdict

    def qSRG(self, axes=None, registers=None):
        """Get register values for queried 'axes' and 'registers'.
        @param axes: Axis or list of them.
        @param registers : Integer convertible or list of them.
        @return : Ordered dictionary of {axis: {register: value}}, axes are str,
        registers/values are integers.
        """
        debug('GCSCommands.qSRG(axes=%r, registers=%r)', axes, registers)
        if registers:
            checksize((), axes, registers)
        cmdstr = self.getcmdstr('SRG?', axes, registers)
        answer = self.__msgs.read(cmdstr)
        answerdict = getdict_twoitems(answer, axes, registers, itemconv=[str, int],
                                      valueconv=(int,))
        debug('GCSCommands.qSRG = %r', answerdict)
        return answerdict

    def qSTA(self):
        """Get register value. Obsolete, use qSRG().
        @return : Register value as integer.
        """
        debug('GCSCommands.qSTA()')
        answer = self.__msgs.read('STA?')
        value = int(answer.strip(), base=16)
        debug('GCSCommands.qSTA = %r', value)
        return value

    def GetStatus(self):
        """Get current position, corresponds to GCS command "#4" which behaves like "SRG?".
        @return Answer as integer.
        """
        debug('GCSCommands.GetStatus()')
        answer = self.__msgs.read(chr(4))
        value = int(answer.strip(), base=16)
        debug('GCSCommands.GetStatus = %r', value)
        return value

    def qWAV(self, tables=None, params=None):
        """Get register values for queried 'tables' and 'params'.
        @param tables: Table ID or list of tables.
        @param params : Integer convertible or list of them.
        @return : Ordered dictionary of {table: {param: value}}, table/param are int, value is
        float.
        """
        debug('GCSCommands.qWAV(tables=%r, params=%r)', tables, params)
        if params:
            checksize((), tables, params)
        cmdstr = self.getcmdstr('WAV?', tables, params)
        answer = self.__msgs.read(cmdstr)
        answerdict = getdict_twoitems(answer, tables, params, itemconv=[int, int], valueconv=(float,))
        debug('GCSCommands.qWAV = %r', answerdict)
        return answerdict

    def qTRA(self, axes, directions=None):
        """Get the maximum absolute position which can be reached from the current position in the
        given 'direction' for the queried axis vector.
        @param axes: Axis or list of axes or dictionary {axis : direction}.
        @param directions : Float convertible or list of them or None.
        @return : Ordered dictionary of {axis: value}, values are float.
        """
        debug('GCSCommands.qTRA(axes=%r, directions=%r)', axes, directions)
        axes, directions = getitemsvaluestuple(axes, directions)
        cmdstr = self.getcmdstr('TRA?', axes, directions)
        answer = self.__msgs.read(cmdstr)
        answerdict = getdict_oneitem(answer, axes, valueconv=(float,))
        debug('GCSCommands.qTRA = %r', answerdict)
        return answerdict

    def qKLC(self, csname1='', csname2='', item1='', item2=''):
        """Get properties of combinations of Work and Tool coordinate systems.
        @param csname1 : String with name of the coordinate system.
        @param csname2 : String with name of the coordinate system.
        @param item1 : String with first item to query.
        @param item2 : String with second item to query.
        @return : String with answer, lines are separated by linefeed characters.
        """
        debug('GCSCommands.qKLC(csname1=%r, csname2=%r, item1=%r, item2=%r)', csname1, csname2, item1, item2)
        if item2:
            checksize((1, 1, 1, 1), csname1, csname2, item1, item2)
        elif item1:
            checksize((1, 1, 1), csname1, csname2, item1)
        elif csname2:
            checksize((1, 1), csname1, csname2)
        elif csname1:
            checksize((1,), csname1)
        cmdstr = self.getcmdstr('KLC?', csname1, csname2, item1, item2)
        answer = self.__msgs.read(cmdstr)
        debug('GCSCommands.qKLC = %r', answer)
        return answer

    def qKLS(self, csname='', item1='', item2=''):
        """Get properties of all coordinate systems.
        @param csname : String with name of the coordinate system.
        @param item1 : String with first item to query.
        @param item2 : String with second item to query.
        @return : String with answer, lines are separated by linefeed characters.
        """
        debug('GCSCommands.qKLS(csname=%r, item1=%r, item2=%r)', csname, item1, item2)
        if item2:
            checksize((1, 1, 1), csname, item1, item2)
        elif item1:
            checksize((1, 1), csname, item1)
        elif csname:
            checksize((1,), csname)
        cmdstr = self.getcmdstr('KLS?', csname, item1, item2)
        answer = self.__msgs.read(cmdstr)
        debug('GCSCommands.qKLS = %r', answer)
        return answer

    def qKLT(self, csstart='', csend=''):
        """Get the position and orientation of the coordinate system which results from a chain of
        linked coordinate systems, or from a part of a chain.
        @param csstart : String with name of the coordinate system at start point of the chain.
        @param csend : String with name of the coordinate system at end point of the chain.
        @return : String with answer, lines are separated by linefeed characters.
        """
        debug('GCSCommands.qKLT(csstart=%r, csend=%r)', csstart, csend)
        if csend:
            checksize((1, 1), csstart, csend)
        elif csstart:
            checksize((1,), csstart)
        cmdstr = self.getcmdstr('KLT?', csstart, csend)
        answer = self.__msgs.read(cmdstr)
        debug('GCSCommands.qKLT = %r', answer)
        return answer

    def qWGS(self, wavegen=None, item=None):
        """Get wave generator status information.
        @param wavegen : Wave generator ID as integer or None for all wave generators.
        @param item : Name of item to query as string or None for all items.
        @return : Ordered dictionary of {wavegen: {item: value}}, wavegen is integer,
        item/value are string.
        """
        debug('GCSCommands.qWGS(wavegen=%r, item=%r)', wavegen, item)
        if wavegen or item:
            checksize((1,), wavegen)
        if item:
            checksize((1,), item)
        cmdstr = self.getcmdstr('WGS?', wavegen, item)
        answer = self.__msgs.read(cmdstr)
        answerdict = getdict_twoitems(answer, wavegen, item, itemconv=[int, str], valueconv=(True,))
        debug('GCSCommands.qWGS = %r', answerdict)
        return answerdict

    def qMAS(self, axes=None):
        """Get the electronic gearing master axes for 'axes'.
        @param axes : String convertible or list of them or None.
        @return : Ordered dictionary {axis: master}, master as string.
        """
        debug('GCSCommands.qMAS(axes=%r)', axes)
        cmdstr = self.getcmdstr('MAS?', axes)
        answer = self.__msgs.read(cmdstr)
        answerdict = getdict_oneitem(answer, axes, valueconv=(str,))
        debug('GCSCommands.qKLC = %r', answerdict)
        return answerdict

    def qHIE(self, devices=None, devaxes=None):
        """Get elongation of HID axis.
        @param devices: Integer convertible or list of them or None.
        @param devaxes : Integer convertible or list of them or None.
        @return : Ordered dictionary of {device: {axis: value}}, device/axis are integers,
        values are float.
        """
        debug('GCSCommands.qHIE(devices=%r, devaxes=%r)', devices, devaxes)
        if devaxes:
            checksize((), devices, devaxes)
        cmdstr = self.getcmdstr('HIE?', devices, devaxes)
        answer = self.__msgs.read(cmdstr)
        answerdict = getdict_twoitems(answer, devices, devaxes, itemconv=[int, int], valueconv=(float,))
        debug('GCSCommands.qHIE = %r', answerdict)
        return answerdict

    def qHIL(self, devices=None, leds=None):
        """Get state of HID LED.
        LED is a hardware dependent output. This can be a LED or a force-feedback actuator.
        @param devices: Integer convertible or list of them or None.
        @param leds : Integer convertible or list of them.
        @return : Ordered dictionary of {device: {led: value}}, all are integers.
        """
        debug('GCSCommands.qHIL(devices=%r, leds=%r)', devices, leds)
        if leds:
            checksize((), devices, leds)
        cmdstr = self.getcmdstr('HIL?', devices, leds)
        answer = self.__msgs.read(cmdstr)
        answerdict = getdict_twoitems(answer, devices, leds, itemconv=[int, int], valueconv=(int,))
        debug('GCSCommands.qHIL = %r', answerdict)
        return answerdict

    def qJAS(self, devices=None, devaxes=None):
        """Get status of joystick axes.
        @param devices: Device ID or list of devices or None.
        @param devaxes : Integer convertible or list of them or None.
        @return : Ordered dictionary of {device: {axis: value}}, device, axes are integers, value is float.
        """
        debug('GCSCommands.qJAS(devices=%r, devaxes=%r)', devices, devaxes)
        if devaxes:
            checksize((), devices, devaxes)
        cmdstr = self.getcmdstr('JAS?', devices, devaxes)
        answer = self.__msgs.read(cmdstr)
        answerdict = getdict_twoitems(answer, devices, devaxes, itemconv=[int, int], valueconv=(float,))
        debug('GCSCommands.qJAS = %r', answerdict)
        return answerdict

    def qJAX(self, devices=None, devaxes=None):
        """Get axes controlled by joystick 'devices' axes 'devaxes'.
        @param devices: Device ID or list of devices or None.
        @param devaxes : Integer convertible or list of them or None.
        @return : Ordered dictionary of {device: {axis: value}}, device/axes are integers,
        values are string.
        """
        debug('GCSCommands.qJAX(devices=%r, devaxes=%r)', devices, devaxes)
        if devaxes:
            checksize((), devices, devaxes)
        cmdstr = self.getcmdstr('JAX?', devices, devaxes)
        answer = self.__msgs.read(cmdstr)
        answerdict = getdict_twoitems(answer, devices, devaxes, itemconv=[int, int], valueconv=(str,))
        debug('GCSCommands.qJAX = %r', answerdict)
        return answerdict

    def qMOD(self, items=None, modes=None):
        """Get modes.
        @param items: Axes/channels/systems or list of them or None.
        @param modes : Integer convertible or list of them or None.
        @return : Ordered dictionary of {item: {mode: value}}, items are string, modes are int,
        values are string.
        """
        debug('GCSCommands.qMOD(items=%r, modes=%r)', items, modes)
        if modes:
            checksize((), items, modes)
        cmdstr = self.getcmdstr('MOD?', items, modes)
        answer = self.__msgs.read(cmdstr)
        answerdict = getdict_twoitems(answer, items, modes, itemconv=[str, int], valueconv=(str,))
        debug('GCSCommands.qMOD = %r', answerdict)
        return answerdict

    def qJBS(self, devices=None, buttons=None):
        """Get the current status of 'buttons' of joystick 'devices'.
        @param devices: Device ID or list of devices or dictionary {device : led}.
        @param buttons : Integer convertible or list of them or None.
        @return : Ordered dictionary of {device: {button: value}}, all are integers.
        """
        debug('GCSCommands.qJBS(devices=%r, buttons=%r)', devices, buttons)
        if buttons:
            checksize((), devices, buttons)
        cmdstr = self.getcmdstr('JBS?', devices, buttons)
        answer = self.__msgs.read(cmdstr)
        answerdict = getdict_twoitems(answer, devices, buttons, itemconv=[int, int], valueconv=(int,))
        debug('GCSCommands.qJBS = %r', answerdict)
        return answerdict

    def qVMO(self, axes, targets=None):
        """Test if move of 'axes' to 'targets' is possible, i.e. withing the current motion range.
        @param axes: Axis or list of axes or dictionary {axis : target}.
        @param targets : Float convertible or list of them or None.
        @return : True if move is possible.
        """
        debug('GCSCommands.qVMO(axes=%r, targets=%r)', axes, targets)
        axes, targets = getitemsvaluestuple(axes, targets)
        cmdstr = self.getcmdstr('VMO?', axes, targets)
        answer = self.__msgs.read(cmdstr)
        value = answer == '1\n'
        debug('GCSCommands.qVMO = %r', value)
        return value

    def qWGI(self, wavegens=None):
        """Get index of the currently output wave point.
        @param wavegens : Wave generator ID as nteger convertible or list of them or None.
        @return : Ordered dictionary of {wavegen: value}, values are int.
        """
        debug('GCSCommands.qWGI(wavegens=%r)', wavegens)
        cmdstr = self.getcmdstr('WGI?', wavegens)
        answer = self.__msgs.read(cmdstr)
        answerdict = getdict_oneitem(answer, wavegens, valueconv=(int,), itemconv=int)
        debug('GCSCommands.qWGI = %r', answerdict)
        return answerdict

    def qWGN(self, wavegens=None):
        """Get number of finished wave cycles since last WGO().
        @param wavegens : Wave generator ID as nteger convertible or list of them or None.
        @return Ordered dictionary of {wavegen: value}, values are int.
        """
        debug('GCSCommands.qWGN(wavegens=%r)', wavegens)
        cmdstr = self.getcmdstr('WGN?', wavegens)
        answer = self.__msgs.read(cmdstr)
        answerdict = getdict_oneitem(answer, wavegens, valueconv=(int,), itemconv=int)
        debug('GCSCommands.qWGN = %r', answerdict)
        return answerdict

    def qWSL(self, wavegens=None):
        """Get current setting of wave table selection.
        @param wavegens : Wave generator ID as nteger convertible or list of them or None.
        @return Ordered dictionary of {wavegen: value}, values are int.
        """
        debug('GCSCommands.qWSL(wavegens=%r)', wavegens)
        cmdstr = self.getcmdstr('WSL?', wavegens)
        answer = self.__msgs.read(cmdstr)
        answerdict = getdict_oneitem(answer, wavegens, valueconv=(int,), itemconv=int)
        debug('GCSCommands.qWSL = %r', answerdict)
        return answerdict

    def qDTL(self, tables=None):
        """Get dynamic digital Linearization (DDL) table length.
        @param tables : Wave table ID as nteger convertible or list of them or None.
        @return Ordered dictionary of {table: value}, values are int.
        """
        debug('GCSCommands.qDTL(tables=%r)', tables)
        cmdstr = self.getcmdstr('DTL?', tables)
        answer = self.__msgs.read(cmdstr)
        answerdict = getdict_oneitem(answer, tables, valueconv=(int,), itemconv=int)
        debug('GCSCommands.qDTL = %r', answerdict)
        return answerdict

    def qONL(self, channels=None):
        """Get current remote control status setting.
        @param channels : Controller channel ID as integer convertible or list of them or None.
        @return Ordered dictionary of {channel: value}, values are bool.
        """
        debug('GCSCommands.qONL(channels=%r)', channels)
        cmdstr = self.getcmdstr('ONL?', channels)
        answer = self.__msgs.read(cmdstr)
        answerdict = getdict_oneitem(answer, channels, valueconv=(bool,), itemconv=int)
        debug('GCSCommands.qONL = %r', answerdict)
        return answerdict

    def qOSN(self, channels=None):
        """Get the number of left steps of last commanded open loop step moving.
        @param channels : Nexline channel ID as integer convertible or list of them or None.
        @return Ordered dictionary of {channel: value}, values are float.
        """
        debug('GCSCommands.qOSN(channels=%r)', channels)
        cmdstr = self.getcmdstr('OSN?', channels)
        answer = self.__msgs.read(cmdstr)
        answerdict = getdict_oneitem(answer, channels, valueconv=(float,), itemconv=int)
        debug('GCSCommands.qOSN = %r', answerdict)
        return answerdict

    def qTRO(self, lines=None):
        """Get trigger output mode enable status for given trigger output line.
        @param lines : Integer convertible or list of them or None.
        @return Ordered dictionary of {line: value}, values are bool.
        """
        debug('GCSCommands.qTRO(lines=%r)', lines)
        cmdstr = self.getcmdstr('TRO?', lines)
        answer = self.__msgs.read(cmdstr)
        answerdict = getdict_oneitem(answer, lines, valueconv=(bool,), itemconv=int)
        debug('GCSCommands.qTRO = %r', answerdict)
        return answerdict

    def qTRI(self, lines=None):
        """Get trigger input mode enable status for given trigger output line.
        @param lines : Integer convertible or list of them or None.
        @return Ordered dictionary of {line: value}, values are bool.
        """
        debug('GCSCommands.qTRI(lines=%r)', lines)
        cmdstr = self.getcmdstr('TRI?', lines)
        answer = self.__msgs.read(cmdstr)
        answerdict = getdict_oneitem(answer, lines, valueconv=(bool,), itemconv=int)
        debug('GCSCommands.qTRI = %r', answerdict)
        return answerdict

    def qJON(self, devices=None):
        """Get joystick enable status.
        @param devices : Joystick device ID as integer convertible or list of them or None.
        @return Ordered dictionary of {device: value}, values are bool.
        """
        debug('GCSCommands.qJON(devices=%r)', devices)
        cmdstr = self.getcmdstr('JON?', devices)
        answer = self.__msgs.read(cmdstr)
        answerdict = getdict_oneitem(answer, devices, valueconv=(bool,), itemconv=int)
        debug('GCSCommands.qJON = %r', answerdict)
        return answerdict

    def qDIP(self, axes=None):
        """Ask if a digital pulse was detected since the last call of qDIP().
        @param axes : String convertible or list of them or None.
        @return : Ordered dictionary of {axis: value}, values are bool.
        """
        debug('GCSCommands.qDIP(axes=%r)', axes)
        cmdstr = self.getcmdstr('DIP?', axes)
        answer = self.__msgs.read(cmdstr)
        answerdict = getdict_oneitem(answer, axes, valueconv=(bool,), itemconv=int)
        debug('GCSCommands.qDIP = %r', answerdict)
        return answerdict

    def qEGE(self, axes=None):
        """Get electronic gearing enable status for 'axes'.
        @param axes : String convertible or list of them or None.
        @return : Ordered dictionary of {axis: value}, values are bool.
        """
        debug('GCSCommands.qEGE(axes=%r)', axes)
        cmdstr = self.getcmdstr('EGE?', axes)
        answer = self.__msgs.read(cmdstr)
        answerdict = getdict_oneitem(answer, axes, valueconv=(bool,))
        debug('GCSCommands.qEGE = %r', answerdict)
        return answerdict

    def qFES(self, axes=None):
        """Get status of search for a signal edge for 'axes'.
        @param axes : String convertible or list of them or None.
        @return : Ordered dictionary of {axis: value}, values are bool.
        """
        debug('GCSCommands.qFES(axes=%r)', axes)
        cmdstr = self.getcmdstr('FES?', axes)
        answer = self.__msgs.read(cmdstr)
        answerdict = getdict_oneitem(answer, axes, valueconv=(bool,))
        debug('GCSCommands.qFES = %r', answerdict)
        return answerdict

    def qFRF(self, axes=None):
        """Indicate whether the given 'axes' are referenced or not.
        An axis is considered as "referenced" when the current position value is set to a known
        position. Depending on the controller, this is the case if a reference move was
        successfully executed with FRF(), FNL() or FPL(), or if the position was set manually
        with POS().
        @param axes : String convertible or list of them or None.
        @return : Ordered dictionary of {axis: value}, values are bool.
        """
        debug('GCSCommands.qFRF(axes=%r)', axes)
        cmdstr = self.getcmdstr('FRF?', axes)
        answer = self.__msgs.read(cmdstr)
        answerdict = getdict_oneitem(answer, axes, valueconv=(bool,))
        debug('GCSCommands.qFRF = %r', answerdict)
        return answerdict

    def qHAR(self, axes=None):
        """Report whether the hard stops of 'axes' can be used for reference moves.
        @param axes : String convertible or list of them or None.
        @return : Ordered dictionary of {axis: value}, values are bool.
        """
        debug('GCSCommands.qHAR(axes=%r)', axes)
        cmdstr = self.getcmdstr('HAR?', axes)
        answer = self.__msgs.read(cmdstr)
        answerdict = getdict_oneitem(answer, axes, valueconv=(bool,))
        debug('GCSCommands.qHAR = %r', answerdict)
        return answerdict

    def qHIN(self, axes=None):
        """Get the activation state of the control by HID devices.
        @param axes : String convertible or list of them or None.
        @return : Ordered dictionary of {axis: value}, values are bool.
        """
        debug('GCSCommands.qHIN(axes=%r)', axes)
        cmdstr = self.getcmdstr('HIN?', axes)
        answer = self.__msgs.read(cmdstr)
        answerdict = getdict_oneitem(answer, axes, valueconv=(bool,))
        debug('GCSCommands.qHIN = %r', answerdict)
        return answerdict

    def qIFC(self, items=None):
        """Get the interface configuration from volatile memory.
        @param items : List of items or single item to query or None to query all.
        @return : Ordered dictionary {item: value} as strings.
        """
        debug('GCSCommands.qIFC(items=%r)', items)
        cmdstr = self.getcmdstr('IFC?', items)
        answer = self.__msgs.read(cmdstr)
        answerdict = getdict_oneitem(answer, items, valueconv=(str,))
        debug('GCSCommands.qIFC = %r', answerdict)
        return answerdict

    def qIFS(self, items=None):
        """Get the interface configuration stored in non-volatile memory.
        @param items : List of items or single item to query or None to query all.
        @return : Ordered dictionary {item: value}.
        """
        debug('GCSCommands.qIFS(items=%r)', items)
        cmdstr = self.getcmdstr('IFS?', items)
        answer = self.__msgs.read(cmdstr)
        answerdict = getdict_oneitem(answer, items, valueconv=(str,))
        debug('GCSCommands.qIFS = %r', answerdict)
        return answerdict

    def qONT(self, axes=None):
        """Check if 'axes' have reached the target.
        @param axes : String convertible or list of them or None.
        @return : Ordered dictionary of {axis: value}, values are bool.
        """
        debug('GCSCommands.qONT(axes=%r)', axes)
        cmdstr = self.getcmdstr('ONT?', axes)
        answer = self.__msgs.read(cmdstr)
        answerdict = getdict_oneitem(answer, axes, valueconv=(bool,))
        debug('GCSCommands.qONT = %r', answerdict)
        return answerdict

    def qOVF(self, axes=None):
        """Check overflow status of 'axes'.
        Overflow means that the control variables are out of range (can only happen if controller
        is in closed-loop mode).
        @param axes : String convertible or list of them or None.
        @return : Ordered dictionary of {axis: value}, values are bool.
        """
        debug('GCSCommands.qOVF(axes=%r)', axes)
        cmdstr = self.getcmdstr('OVF?', axes)
        answer = self.__msgs.read(cmdstr)
        answerdict = getdict_oneitem(answer, axes, valueconv=(bool,))
        debug('GCSCommands.qOVF = %r', answerdict)
        return answerdict

    def qMAN(self, item=''):
        """Show detailed help for 'item'.
        @param item : GCS command to get help for.
        @return : Help message as string with trailing linefeed.
        """
        debug('GCSCommands.qMAN()')
        if item:
            checksize((1,), item)
        cmdstr = self.getcmdstr('MAN?', item)
        answer = self.__msgs.read(cmdstr)
        debug('GCSCommands.qMAN = %r', answer)
        return answer

    def qMAC(self, item=''):
        """Get available macros, or list contents of a specific macro.
        If 'item' is empty, all available macros are listed,
        separated with line-feed characters. Otherwise the content of the
        macro 'item' is listed, the single lines separated by line-feed
        characters. If there are no macros stored or the requested macro is
        empty the answer will be "".
        @param item : Optional name of macro to list.
        @return : String.
        """
        debug('GCSCommands.qMAC()')
        if item:
            checksize((1,), item)
        cmdstr = self.getcmdstr('MAC?', item)
        answer = self.__msgs.read(cmdstr)
        debug('GCSCommands.qMAC = %r', answer)
        return answer

    def qHPA(self):
        """Get the help string about available parameters with short descriptions.
        @return : Answer as string with trailing linefeed.
        """
        debug('GCSCommands.qHPA()')
        answer = self.__msgs.read('HPA?')
        debug('GCSCommands.qHPA = %r', answer)
        return answer

    def qHDR(self):
        """Lists a help string for data recording.
        Contains record options and trigger options, information about
        additional parameters and commands regarding data recording.
        @return : Answer as string with trailing linefeed.
        """
        debug('GCSCommands.qHDR()')
        answer = self.__msgs.read('HDR?')
        debug('GCSCommands.qHDR = %r', answer)
        return answer

    def qRMC(self):
        """List macros which are currently running.
        @return : Answer as string with trailing linefeed.
        """
        debug('GCSCommands.qRMC()')
        answer = self.__msgs.read('RMC?')
        debug('GCSCommands.qRMC = %r', answer)
        return answer

    def qREF(self, axes=None):
        """Check if the given 'axes' have a reference.
        @param axes : String convertible or list of them or None.
        @return : Ordered dictionary of {axis: value}, values are bool.
        """
        debug('GCSCommands.qREF(axes=%r)', axes)
        cmdstr = self.getcmdstr('REF?', axes)
        answer = self.__msgs.read(cmdstr)
        answerdict = getdict_oneitem(answer, axes, valueconv=(bool,))
        debug('GCSCommands.qREF = %r', answerdict)
        return answerdict

    def qRON(self, axes=None):
        """Get reference mode for given 'axes'.
        @param axes : String convertible or list of them or None.
        @return : Ordered dictionary of {axis: value}, values are bool.
        """
        debug('GCSCommands.qRON(axes=%r)', axes)
        cmdstr = self.getcmdstr('RON?', axes)
        answer = self.__msgs.read(cmdstr)
        answerdict = getdict_oneitem(answer, axes, valueconv=(bool,))
        debug('GCSCommands.qRON = %r', answerdict)
        return answerdict

    def qRTO(self, axes=None):
        """Get the "ready-for-turn-off state" of the given 'axes'.
        @param axes : String convertible or list of them or None.
        @return : Ordered dictionary of {axis: value}, values are bool.
        """
        debug('GCSCommands.qRTO(axes=%r)', axes)
        cmdstr = self.getcmdstr('RTO?', axes)
        answer = self.__msgs.read(cmdstr)
        answerdict = getdict_oneitem(answer, axes, valueconv=(bool,))
        debug('GCSCommands.qRTO = %r', answerdict)
        return answerdict

    def qSAI_ALL(self):
        """Get the identifiers for all axes (configured and non configured axes).
        Call qSAI() to only get the activated axes.
        @return : List of all axes as string, whitespaces are removed.
        """
        debug('GCSCommands.qSAI_ALL()')
        answer = self.__msgs.read('SAI? ALL')
        answer = splitparams(answer, self.isgcs2)
        debug('GCSCommands.qSAI_ALL = %r', answer)
        return answer

    def qSSL(self, axes=None):
        """Get the state of the soft limits.
        Limits are set with NLM() and PLM() for 'axes'. If all arguments are omitted, the state is
        queried for all axes.
        @param axes : String convertible or list of them or None.
        @return : Ordered dictionary of {axis: value}, values are bool.
        """
        debug('GCSCommands.qSSL(axes=%r)', axes)
        cmdstr = self.getcmdstr('SSL?', axes)
        answer = self.__msgs.read(cmdstr)
        answerdict = getdict_oneitem(answer, axes, valueconv=(bool,))
        debug('GCSCommands.qSSL = %r', answerdict)
        return answerdict

    def qTRS(self, axes=None):
        """Ask if 'axes' have reference sensors with direction sensing.
        @param axes : String convertible or list of them or None.
        @return : Ordered dictionary of {axis: value}, values are bool.
        """
        debug('GCSCommands.qTRS(axes=%r)', axes)
        cmdstr = self.getcmdstr('TRS?', axes)
        answer = self.__msgs.read(cmdstr)
        answerdict = getdict_oneitem(answer, axes, valueconv=(bool,))
        debug('GCSCommands.qTRS = %r', answerdict)
        return answerdict

    def qVCO(self, axes=None):
        """Get the velocity-control mode for 'axes'.
        @param axes : String convertible or list of them or None.
        @return : Ordered dictionary of {axis: value}, values are bool.
        """
        debug('GCSCommands.qVCO(axes=%r)', axes)
        cmdstr = self.getcmdstr('VCO?', axes)
        answer = self.__msgs.read(cmdstr)
        answerdict = getdict_oneitem(answer, axes, valueconv=(bool,))
        debug('GCSCommands.qVCO = %r', answerdict)
        return answerdict

    def DIO(self, lines, values=None):
        """Set digital output 'lines' HIGH or LOW.
        @param lines: Lines as int or string or list of them or dictionary {line : value}.
        @param values : Bool convertible or list of them or None.
        """
        debug('GCSCommands.DIO(lines=%r, values=%r)', lines, values)
        lines, values = getitemsvaluestuple(lines, values)
        cmdstr = self.getcmdstr('DIO', lines, values)
        self.__msgs.send(cmdstr)

    def qDIO(self, lines=None):
        """Get the states of the specified digital input 'lines'.
        Use qTIO() to get the number of installed digital I/O lines.
        @param lines : Lines as int or list of them or None.
        @return : Ordered dictionary of {line: value}, lines are int, values are bool.
        """
        debug('GCSCommands.qDIO(lines=%r)', lines)
        cmdstr = self.getcmdstr('DIO?', lines)
        answer = self.__msgs.read(cmdstr)
        answerdict = getdict_oneitem(answer, lines, valueconv=(bool,), itemconv=int)
        debug('GCSCommands.qDIO = %r', answerdict)
        return answerdict

    def SGP(self, grouptype, groupindex, moduletype, moduleindex, parameter, value):
        """Set firmware parameters classified by groups and mopdules.
        @param grouptype : Group type as string convertible or list of them, e.g. "axes".
        @param groupindex : Enumerates all items within 'grouptype' as int or list of them.
        @param moduletype : Module type as string convertible or list of them, e.g. "notchfilter".
        @param moduleindex : Enumerates all items within 'moduletype' as int or list of them.
        @param parameter : Parameter ID to set as integer or list of them.
        @param value : Value to set 'parameter' to as string convertible or list of them.
        """
        debug('GCSCommands.SGP(grouptype=%r, groupindex=%r, moduletype=%r, moduleindex=%r, parameter=%r, value=%r)',
              grouptype, groupindex, moduletype, moduleindex, parameter, value)
        checksize((True,), grouptype, groupindex, moduletype, moduleindex, parameter, value)
        cmdstr = self.getcmdstr('SGP', grouptype, groupindex, moduletype, moduleindex, parameter, value)
        self.__msgs.send(cmdstr)

    def qSGP(self, grouptype=None, groupindex=None, moduletype=None, moduleindex=None, parameter=None):
        """Query firmware parameters classified by groups and mopdules.
        @param grouptype : Group type as string convertible or list of them, e.g. "axes".
        @param groupindex : Enumerates all items within 'grouptype' as int or list of them.
        @param moduletype : Module type as string convertible or list of them, e.g. "notchfilter".
        @param moduleindex : Enumerates all items within 'moduletype' as int or list of them.
        @param parameter : Parameter ID to set as integer or list of them.
        @return : Tuple of (grouptype (str), groupindex (int), moduletype (str), moduleindex(int), parameter (int),
        value), value is converted accordingly.
        """
        debug('GCSCommands.qSGP(grouptype=%r, groupindex=%r, moduletype=%r, moduleindex=%r, parameter=%r)',
              grouptype, groupindex, moduletype, moduleindex, parameter)
        if parameter:
            checksize((True, True, True, True, True), grouptype, groupindex, moduletype, moduleindex, parameter)
        elif moduleindex:
            checksize((True, True, True, True), grouptype, groupindex, moduletype, moduleindex)
        elif groupindex:
            checksize((True, True), grouptype, groupindex)
            if moduletype:
                raise SystemError('parameter size mismatch: moduletype')
        else:
            if grouptype:
                raise SystemError('parameter size mismatch: grouptype')
        cmdstr = self.getcmdstr('SGP?', grouptype, groupindex, moduletype, moduleindex, parameter)
        answer = self.__msgs.read(cmdstr)
        grouptype, groupindex, moduletype, moduleindex, parameter, value = [], [], [], [], [], []
        for line in answer.splitlines():
            value.append(convertvalue(line.split('=')[1].strip(), totype=True))
            grouptype.append(str(line.split('=')[0].split()[0].strip()))
            groupindex.append(int(line.split('=')[0].split()[1].strip()))
            moduletype.append(str(line.split('=')[0].split()[2].strip()))
            moduleindex.append(int(line.split('=')[0].split()[3].strip()))
            parameter.append(int(line.split('=')[0].split()[4].strip(), base=0))
        answer = (grouptype, groupindex, moduletype, moduleindex, parameter, value)
        debug('GCSCommands.qSGP = %r', answer)
        return answer

    def SPA(self, items, params=None, values=None):
        """Set specified parameters 'params' for 'items' in RAM to 'values'.
        @param items : Axes/channels/systems as string convertible or list of them or dict {item : {param : value}}.
        @param params : Parameter ID as integer convertible or list of them or None if 'items' is a dictionary.
        @param values : Parameter value to set as string convertible or list of them or None if 'items' is a
        dictionary {item : {param : value}}. True/False is not allowed, use 1/0 instead.
        """
        items, params, values = getitemsparamsvaluestuple(items, params, values)
        paramstr = gethexstr(params)
        debug('GCSCommands.SPA(items=%r, params=%s, values=%r)', items, paramstr, values)
        cmdstr = self.getcmdstr('SPA', items, params, values)
        self.__msgs.send(cmdstr)

    def SEP(self, password, items, params=None, values=None):
        """Set specified parameters 'params' for 'items' in non-volatile memory to 'values'.
        @param password : String convertible, usually "100".
        @param items : Axes/channels/systems as string convertible or list of them or dict {item : {param : value}}.
        @param params : Parameter ID as integer convertible or list of them or None if 'items' is a dictionary..
        @param values : Parameter value to set as string convertible or list of them or None if 'items' is a
        dictionary {item : {param : value}}. True/False is not allowed, use 1/0 instead.
        """
        items, params, values = getitemsparamsvaluestuple(items, params, values)
        checksize((1,), password)
        paramstr = gethexstr(params)
        debug('GCSCommands.SEP(password=%r, items=%r, params=%s, values=%r)', password, items, paramstr, values)
        cmdstr = self.getcmdstr('SEP', password, items, params, values)
        self.__msgs.send(cmdstr)

    def qSPA(self, items=None, params=None):
        """Query specified parameters 'params' for 'items' from RAM.
        @param items: Item or list of items or None or dictionary of {item : param}.
        @param params : Integer convertible or list of them or None. Required if 'items' is not a dict.
        @return : Ordered dictionary of {item: {param: value}}, items are string, params are int,
        values are converted to an according type (string, int, float).
        Hint: To get the value of one queried parameter you can use e.g. qSPA('X', 0x123)['X'][0x123].
        """
        items, params = getitemsvaluestuple(items, params, required=False)
        paramstr = gethexstr(params)
        debug('GCSCommands.qSPA(items=%r, params=%s)', items, paramstr)
        if items:
            checksize((True,), items, params)
        cmdstr = self.getcmdstr('SPA?', items, params)
        answer = self.__msgs.read(cmdstr)
        answerdict = getdict_twoitems(answer, items, params, itemconv=[str, int], valueconv=(None,),
                                      convlisttostring=True)
        answerdict = self.paramconv(answerdict)
        debug('GCSCommands.qSPA = %r', answerdict)
        return answerdict

    def qSEP(self, items=None, params=None):
        """Query specified parameters 'params' for 'items' from non-volatile memory.
        @param items: Item or list of items or None or dictionary of {item : param}.
        @param params : Integer convertible or list of them or None. Required if 'items' is not a dict.
        @return : Ordered dictionary of {item: {param: value}}, items are string, params are int,
        values are converted to an according type (string, int, float).
        Hint: To get the value of one queried parameter you can use e.g. qSEP('X', 0x123)['X'][0x123].
        """
        items, params = getitemsvaluestuple(items, params, required=False)
        paramstr = gethexstr(params)
        debug('GCSCommands.qSEP(items=%r, params=%s)', items, paramstr)
        if items:
            checksize((True,), items, params)
        cmdstr = self.getcmdstr('SEP?', items, params)
        answer = self.__msgs.read(cmdstr)
        answerdict = getdict_twoitems(answer, items, params, itemconv=[str, int], valueconv=(None,),
                                      convlisttostring=True)
        answerdict = self.paramconv(answerdict)
        debug('GCSCommands.qSEP = %r', answerdict)
        return answerdict

    def qLST(self):
        """Get names of data files in non-volatile memory.
        @return : Data file names as string with trailing linefeed.
        """
        debug('GCSCommands.qLST()')
        answer = self.__msgs.read('LST?')
        debug('GCSCommands.qLST = %r', answer)
        return answer

    def POL(self, axes, values=None):
        """Set Axis to either 0% or 100% of his maximum range. This depends on the POLarization.
        @param axes: Axis or list of axes or dictionary {axis : value}.
        @param values : A List of 0 and 1. 0 for 0% and 1 for 100%.
        """
        debug('GCSCommands.POL(axes=%r, values=%r)', axes, values)
        axes, values = getitemsvaluestuple(axes, values)
        cmdstr = self.getcmdstr('POL', axes, values)
        self.__msgs.send(cmdstr)

    def STD(self, tabletype, tableid, data):
        """Saves the content of a given table from volatile memory to a data file in nonvolatile memory.
        @param tabletype : Is the type of the table whose data is to be saved.
        @param tableid : Is the identifier of the table whose data is to be saved.
        @param data : Is a string.
        """
        debug('GCSCommands.STD(device=%r, devaxis=%r, axes=%r)', tabletype, tableid, data)
        checksize((1, 1, 1), tabletype, tableid, data)
        cmdstr = self.getcmdstr('STD', tabletype, tableid, data)
        self.__msgs.send(cmdstr)

    def RTD(self, tabletype, tableid, name):
        """Reads the content of a data file in nonvolatile memory and writes it to a given table in volatile memory.
        @param tabletype : Is the type of the table to which the data is to be written.
        @param tableid : Is the identifier of the table to which the data is to be written..
        @param name : Is the file name. String consisting of max. 32 characters. Spaces are not allowed.
        """
        debug('GCSCommands.RTD(device=%r, devaxis=%r, axes=%r)', tabletype, tableid, name)
        checksize((1, 1, 1), tabletype, tableid, name)
        cmdstr = self.getcmdstr('RTD', tabletype, tableid, name)
        self.__msgs.send(cmdstr)

    def qRTD(self, tabletype=None, tableid=None, infoid=None):
        """Reads the content of a data table in volatile memory.
        @param tabletype: Is the type of the table from which the data is to be read.
        table type or list of tabel types or None or dictionary of {(tabletype, tableid) : [infoid,],}}.
        @param tableid : Is the identifier of the table from which the data is to be read Integer convertible or
        list of them or None. Required if 'tabletype' is not a dict.
        @param infoid : Is the identifier of the information to be read. Integer convertible or list of them or None.
        Required if 'tabletype' is not a dict.
        @return : Ordered dictionary of {(tabletype, tableid): [infoid, value]}, items are string, params are int,
        values are converted to an according type (string, int, float).
        """
        if tabletype:
            tabletype, tableid, infoid = getitemsparamsidstuple(tabletype, tableid, infoid)

        debug('GCSCommands.qRTD(tabletype=%r, tableid=%r, infoid=%r)', tabletype, tableid, infoid)
        if tableid:
            checksize((1, 1, True), tabletype, tableid, infoid)
        cmdstr = self.getcmdstr('RTD?', tabletype, tableid, infoid)
        answer = self.__msgs.read(cmdstr)
        answerdict = getdict_threeitems(answer, tabletype, tableid, infoid, itemconv=[int, int, int], valueconv=(None,))
        debug('GCSCommands.qRTD = %r', answerdict)
        return answerdict

    def DLT(self, name):
        """Deletes a data file that was saved in non-volatile memory.
        @param name : > is the file name. String consisting of max. 32 characters. Spaces are not allowed.
        """
        debug('GCSCommands.DLT(item=%r)', name)
        checksize((1,), name)
        cmdstr = self.getcmdstr('DLT', name)
        self.__msgs.send(cmdstr)

    # CODEGEN BEGIN ### DO NOT MODIFY THIS LINE !!! ###############################################

    def HasFDG(self):
        """Return True if FDG() is available."""
        return self.__has('FDG')

    def HasFDR(self):
        """Return True if FDR() is available."""
        return self.__has('FDR')

    def HasFGC(self):
        """Return True if FGC() is available."""
        return self.__has('FGC')

    def HasSIC(self):
        """Return True if SIC() is available."""
        return self.__has('SIC')

    def HasFRC(self):
        """Return True if FRC() is available."""
        return self.__has('FRC')

    def HasFRS(self):
        """Return True if FRS() is available."""
        return self.__has('FRS')

    def HasFRP(self):
        """Return True if FRP() is available."""
        return self.__has('FRP')

    def HasqFRP(self):
        """Return True if qFRP() is available."""
        return self.__has('qFRP')

    def HasqFRC(self):
        """Return True if qFRC() is available."""
        return self.__has('qFRC')

    def HasqFGC(self):
        """Return True if qFGC() is available."""
        return self.__has('qFGC')

    def HasqFSF(self):
        """Return True if qFSF() is available."""
        return self.__has('qFSF')

    def HasqSIC(self):
        """Return True if qSIC() is available."""
        return self.__has('qSIC')

    def HasqFRR(self):
        """Return True if qFRR() is available."""
        return self.__has('qFRR')

    def HasqTCI(self):
        """Return True if qTCI() is available."""
        return self.__has('qTCI')

    def HasqFRH(self):
        """Return True if qFRH() is available."""
        return self.__has('qFRH')

    def HasqSAI(self):
        """Return True if qSAI() is available."""
        return self.__has('qSAI')

    def HasqHLP(self):
        """Return True if qHLP() is available."""
        return self.__has('qHLP')

    def HasqPOS(self):
        """Return True if qPOS() is available."""
        return self.__has('qPOS')

    def HasSVO(self):
        """Return True if SVO() is available."""
        return self.__has('SVO')

    def HasqSVO(self):
        """Return True if qSVO() is available."""
        return self.__has('qSVO')

    def HasqFSR(self):
        """Return True if qFSR() is available."""
        return self.__has('qFSR')

    def HasVAR(self):
        """Return True if VAR() is available."""
        return self.__has('VAR')

    def HasVCO(self):
        """Return True if VCO() is available."""
        return self.__has('VCO')

    def HasSPI(self):
        """Return True if SPI() is available."""
        return self.__has('SPI')

    def HasSRA(self):
        """Return True if SRA() is available."""
        return self.__has('SRA')

    def HasSSL(self):
        """Return True if SSL() is available."""
        return self.__has('SSL')

    def HasRON(self):
        """Return True if RON() is available."""
        return self.__has('RON')

    def HasKLD(self):
        """Return True if KLD() is available."""
        return self.__has('KLD')

    def HasKSB(self):
        """Return True if KSB() is available."""
        return self.__has('KSB')

    def HasKSD(self):
        """Return True if KSD() is available."""
        return self.__has('KSD')

    def HasKST(self):
        """Return True if KST() is available."""
        return self.__has('KST')

    def HasKSW(self):
        """Return True if KSW() is available."""
        return self.__has('KSW')

    def HasMAT(self):
        """Return True if MAT() is available."""
        return self.__has('MAT')

    def HasFSF(self):
        """Return True if FSF() is available."""
        return self.__has('FSF')

    def HasMAC_START(self):
        """Return True if MAC_START() is available."""
        return self.__has('MAC_START')

    def HasMAC_BEG(self):
        """Return True if MAC_BEG() is available."""
        return self.__has('MAC_BEG')

    def HasMAC_STOP(self):
        """Return True if MAC_STOP() is available."""
        return self.__has('MAC_STOP')

    def HasMAC_DEL(self):
        """Return True if MAC_DEL() is available."""
        return self.__has('MAC_DEL')

    def HasMEX(self):
        """Return True if MEX() is available."""
        return self.__has('MEX')

    def HasKSF(self):
        """Return True if KSF() is available."""
        return self.__has('KSF')

    def HasKEN(self):
        """Return True if KEN() is available."""
        return self.__has('KEN')

    def HasKRM(self):
        """Return True if KRM() is available."""
        return self.__has('KRM')

    def HasKLF(self):
        """Return True if KLF() is available."""
        return self.__has('KLF')

    def HasINI(self):
        """Return True if INI() is available."""
        return self.__has('INI')

    def HasIsMoving(self):
        """Return True if IsMoving() is available."""
        return self.__has('IsMoving')

    def HasIsGeneratorRunning(self):
        """Return True if IsGeneratorRunning() is available."""
        return self.__has('IsGeneratorRunning')

    def HasGetDynamicMoveBufferSize(self):
        """Return True if GetDynamicMoveBufferSize() is available."""
        return self.__has('GetDynamicMoveBufferSize')

    def HasKCP(self):
        """Return True if KCP() is available."""
        return self.__has('KCP')

    def HasKLN(self):
        """Return True if KLN() is available."""
        return self.__has('KLN')

    def HasHLT(self):
        """Return True if HLT() is available."""
        return self.__has('HLT')

    def HasIFC(self):
        """Return True if IFC() is available."""
        return self.__has('IFC')

    def HasWGC(self):
        """Return True if WGC() is available."""
        return self.__has('WGC')

    def HasWGO(self):
        """Return True if WGO() is available."""
        return self.__has('WGO')

    def HasWMS(self):
        """Return True if WMS() is available."""
        return self.__has('WMS')

    def HasIFS(self):
        """Return True if IFS() is available."""
        return self.__has('IFS')

    def HasWPA(self):
        """Return True if WPA() is available."""
        return self.__has('WPA')

    def HasDPA(self):
        """Return True if DPA() is available."""
        return self.__has('DPA')

    def HasHasPosChanged(self):
        """Return True if HasPosChanged() is available."""
        return self.__has('HasPosChanged')

    def HasHIN(self):
        """Return True if HIN() is available."""
        return self.__has('HIN')

    def HasGOH(self):
        """Return True if GOH() is available."""
        return self.__has('GOH')

    def HasFED(self):
        """Return True if FED() is available."""
        return self.__has('FED')

    def HasFNL(self):
        """Return True if FNL() is available."""
        return self.__has('FNL')

    def HasFPH(self):
        """Return True if FPH() is available."""
        return self.__has('FPH')

    def HasFPL(self):
        """Return True if FPL() is available."""
        return self.__has('FPL')

    def HasFRF(self):
        """Return True if FRF() is available."""
        return self.__has('FRF')

    def HasDPO(self):
        """Return True if DPO() is available."""
        return self.__has('DPO')

    def HasTRI(self):
        """Return True if TRI() is available."""
        return self.__has('TRI')

    def HasTRO(self):
        """Return True if TRO() is available."""
        return self.__has('TRO')

    def HasTSP(self):
        """Return True if TSP() is available."""
        return self.__has('TSP')

    def HasWSL(self):
        """Return True if WSL() is available."""
        return self.__has('WSL')

    def HasDFH(self):
        """Return True if DFH() is available."""
        return self.__has('DFH')

    def HasDCO(self):
        """Return True if DCO() is available."""
        return self.__has('DCO')

    def HasEAX(self):
        """Return True if EAX() is available."""
        return self.__has('EAX')

    def HasATZ(self):
        """Return True if ATZ() is available."""
        return self.__has('ATZ')

    def HasCTI(self):
        """Return True if CTI() is available."""
        return self.__has('CTI')

    def HasDDL(self):
        """Return True if DDL() is available."""
        return self.__has('DDL')

    def HasDRT(self):
        """Return True if DRT() is available."""
        return self.__has('DRT')

    def HasWTR(self):
        """Return True if WTR() is available."""
        return self.__has('WTR')

    def HasTWS(self):
        """Return True if TWS() is available."""
        return self.__has('TWS')

    def HasMAC_NSTART(self):
        """Return True if MAC_NSTART() is available."""
        return self.__has('MAC_NSTART')

    def HasMAC_qDEF(self):
        """Return True if MAC_qDEF() is available."""
        return self.__has('MAC_qDEF')

    def HasIsRunningMacro(self):
        """Return True if IsRunningMacro() is available."""
        return self.__has('IsRunningMacro')

    def HasIsControllerReady(self):
        """Return True if IsControllerReady() is available."""
        return self.__has('IsControllerReady')

    def HasMAC_qERR(self):
        """Return True if MAC_qERR() is available."""
        return self.__has('MAC_qERR')

    def HasHDT(self):
        """Return True if HDT() is available."""
        return self.__has('HDT')

    def HasTWE(self):
        """Return True if TWE() is available."""
        return self.__has('TWE')

    def HasHIL(self):
        """Return True if HIL() is available."""
        return self.__has('HIL')

    def HasHIS(self):
        """Return True if HIS() is available."""
        return self.__has('HIS')

    def HasHIT(self):
        """Return True if HIT() is available."""
        return self.__has('HIT')

    def HasJDT(self):
        """Return True if JDT() is available."""
        return self.__has('JDT')

    def HasJAX(self):
        """Return True if JAX() is available."""
        return self.__has('JAX')

    def HasJON(self):
        """Return True if JON() is available."""
        return self.__has('JON')

    def HasHIA(self):
        """Return True if HIA() is available."""
        return self.__has('HIA')

    def HasWAV_NOISE(self):
        """Return True if WAV_NOISE() is available."""
        return self.__has('WAV_NOISE')

    def HasMOD(self):
        """Return True if MOD() is available."""
        return self.__has('MOD')

    def HasSWT(self):
        """Return True if SWT() is available."""
        return self.__has('SWT')

    def HasWTO(self):
        """Return True if WTO() is available."""
        return self.__has('WTO')

    def HasMNL(self):
        """Return True if MNL() is available."""
        return self.__has('MNL')

    def HasMPL(self):
        """Return True if MPL() is available."""
        return self.__has('MPL')

    def HasRST(self):
        """Return True if RST() is available."""
        return self.__has('RST')

    def HasITD(self):
        """Return True if ITD() is available."""
        return self.__has('ITD')

    def HasRTO(self):
        """Return True if RTO() is available."""
        return self.__has('RTO')

    def HasSCH(self):
        """Return True if SCH() is available."""
        return self.__has('SCH')

    def HasSTP(self):
        """Return True if STP() is available."""
        return self.__has('STP')

    def HasTWC(self):
        """Return True if TWC() is available."""
        return self.__has('TWC')

    def HasWGR(self):
        """Return True if WGR() is available."""
        return self.__has('WGR')

    def HasMAC_END(self):
        """Return True if MAC_END() is available."""
        return self.__has('MAC_END')

    def HasStopAll(self):
        """Return True if StopAll() is available."""
        return self.__has('StopAll')

    def HasSystemAbort(self):
        """Return True if SystemAbort() is available."""
        return self.__has('SystemAbort')

    def HasRTR(self):
        """Return True if RTR() is available."""
        return self.__has('RTR')

    def HasDEL(self):
        """Return True if DEL() is available."""
        return self.__has('DEL')

    def HasSAV(self):
        """Return True if SAV() is available."""
        return self.__has('SAV')

    def HasFLM(self):
        """Return True if FLM() is available."""
        return self.__has('FLM')

    def HasFLS(self):
        """Return True if FLS() is available."""
        return self.__has('FLS')

    def HasACC(self):
        """Return True if ACC() is available."""
        return self.__has('ACC')

    def HasADD(self):
        """Return True if ADD() is available."""
        return self.__has('ADD')

    def HasDEC(self):
        """Return True if DEC() is available."""
        return self.__has('DEC')

    def HasDFF(self):
        """Return True if DFF() is available."""
        return self.__has('DFF')

    def HasOAC(self):
        """Return True if OAC() is available."""
        return self.__has('OAC')

    def HasOAD(self):
        """Return True if OAD() is available."""
        return self.__has('OAD')

    def HasODC(self):
        """Return True if ODC() is available."""
        return self.__has('ODC')

    def HasOSM(self):
        """Return True if OSM() is available."""
        return self.__has('OSM')

    def HasOVL(self):
        """Return True if OVL() is available."""
        return self.__has('OVL')

    def HasPOS(self):
        """Return True if POS() is available."""
        return self.__has('POS')

    def HasqACC(self):
        """Return True if qACC() is available."""
        return self.__has('qACC')

    def HasqAOS(self):
        """Return True if qAOS() is available."""
        return self.__has('qAOS')

    def HasqCAV(self):
        """Return True if qCAV() is available."""
        return self.__has('qCAV')

    def HasqCCV(self):
        """Return True if qCCV() is available."""
        return self.__has('qCCV')

    def HasqCMN(self):
        """Return True if qCMN() is available."""
        return self.__has('qCMN')

    def HasqCMX(self):
        """Return True if qCMX() is available."""
        return self.__has('qCMX')

    def HasqCOV(self):
        """Return True if qCOV() is available."""
        return self.__has('qCOV')

    def HasqATC(self):
        """Return True if qATC() is available."""
        return self.__has('qATC')

    def HasqNAV(self):
        """Return True if qNAV() is available."""
        return self.__has('qNAV')

    def HasqTAD(self):
        """Return True if qTAD() is available."""
        return self.__has('qTAD')

    def HasqTAV(self):
        """Return True if qTAV() is available."""
        return self.__has('qTAV')

    def HasqTNS(self):
        """Return True if qTNS() is available."""
        return self.__has('qTNS')

    def HasqTSP(self):
        """Return True if qTSP() is available."""
        return self.__has('qTSP')

    def HasqVOL(self):
        """Return True if qVOL() is available."""
        return self.__has('qVOL')

    def HasqSGA(self):
        """Return True if qSGA() is available."""
        return self.__has('qSGA')

    def HasqDEC(self):
        """Return True if qDEC() is available."""
        return self.__has('qDEC')

    def HasqFPH(self):
        """Return True if qFPH() is available."""
        return self.__has('qFPH')

    def HasqDFF(self):
        """Return True if qDFF() is available."""
        return self.__has('qDFF')

    def HasqDFH(self):
        """Return True if qDFH() is available."""
        return self.__has('qDFH')

    def HasqMOV(self):
        """Return True if qMOV() is available."""
        return self.__has('qMOV')

    def HasqJOG(self):
        """Return True if qJOG() is available."""
        return self.__has('qJOG')

    def HasqNLM(self):
        """Return True if qNLM() is available."""
        return self.__has('qNLM')

    def HasqOAC(self):
        """Return True if qOAC() is available."""
        return self.__has('qOAC')

    def HasqOAD(self):
        """Return True if qOAD() is available."""
        return self.__has('qOAD')

    def HasqOCD(self):
        """Return True if qOCD() is available."""
        return self.__has('qOCD')

    def HasqDRR(self):
        """Return True if qDRR() is available."""
        return self.__has('qDRR')

    def HasqGFR(self):
        """Return True if qGFR() is available."""
        return self.__has('qGFR')

    def HasqDDL(self):
        """Return True if qDDL() is available."""
        return self.__has('qDDL')

    def HasqGWD(self):
        """Return True if qGWD() is available."""
        return self.__has('qGWD')

    def HasqHIT(self):
        """Return True if qHIT() is available."""
        return self.__has('qHIT')

    def HasqJLT(self):
        """Return True if qJLT() is available."""
        return self.__has('qJLT')

    def HasqTWS(self):
        """Return True if qTWS() is available."""
        return self.__has('qTWS')

    def HasqODC(self):
        """Return True if qODC() is available."""
        return self.__has('qODC')

    def HasqOSM(self):
        """Return True if qOSM() is available."""
        return self.__has('qOSM')

    def HasqOVL(self):
        """Return True if qOVL() is available."""
        return self.__has('qOVL')

    def HasqPLM(self):
        """Return True if qPLM() is available."""
        return self.__has('qPLM')

    def HasqSPI(self):
        """Return True if qSPI() is available."""
        return self.__has('qSPI')

    def HasqSSA(self):
        """Return True if qSSA() is available."""
        return self.__has('qSSA')

    def HasqSST(self):
        """Return True if qSST() is available."""
        return self.__has('qSST')

    def HasqSVA(self):
        """Return True if qSVA() is available."""
        return self.__has('qSVA')

    def HasqTCV(self):
        """Return True if qTCV() is available."""
        return self.__has('qTCV')

    def HasqTMN(self):
        """Return True if qTMN() is available."""
        return self.__has('qTMN')

    def HasqTMX(self):
        """Return True if qTMX() is available."""
        return self.__has('qTMX')

    def HasqVEL(self):
        """Return True if qVEL() is available."""
        return self.__has('qVEL')

    def HasqVMA(self):
        """Return True if qVMA() is available."""
        return self.__has('qVMA')

    def HasqVMI(self):
        """Return True if qVMI() is available."""
        return self.__has('qVMI')

    def HasqWOS(self):
        """Return True if qWOS() is available."""
        return self.__has('qWOS')

    def HasRNP(self):
        """Return True if RNP() is available."""
        return self.__has('RNP')

    def HasSSA(self):
        """Return True if SSA() is available."""
        return self.__has('SSA')

    def HasSST(self):
        """Return True if SST() is available."""
        return self.__has('SST')

    def HasVEL(self):
        """Return True if VEL() is available."""
        return self.__has('VEL')

    def HasWOS(self):
        """Return True if WOS() is available."""
        return self.__has('WOS')

    def HasVLS(self):
        """Return True if VLS() is available."""
        return self.__has('VLS')

    def HasTIM(self):
        """Return True if TIM() is available."""
        return self.__has('TIM')

    def HasSCT(self):
        """Return True if SCT() is available."""
        return self.__has('SCT')

    def HasAOS(self):
        """Return True if AOS() is available."""
        return self.__has('AOS')

    def HasVOL(self):
        """Return True if VOL() is available."""
        return self.__has('VOL')

    def HasCST(self):
        """Return True if CST() is available."""
        return self.__has('CST')

    def HasCTR(self):
        """Return True if CTR() is available."""
        return self.__has('CTR')

    def HasCTV(self):
        """Return True if CTV() is available."""
        return self.__has('CTV')

    def HasDMOV(self):
        """Return True if DMOV() is available."""
        return self.__has('DMOV')

    def HasGetPosStatus(self):
        """Return True if GetPosStatus() is available."""
        return self.__has('GetPosStatus')

    def HasMAS(self):
        """Return True if MAS() is available."""
        return self.__has('MAS')

    def HasMOV(self):
        """Return True if MOV() is available."""
        return self.__has('MOV')

    def HasPUN(self):
        """Return True if PUN() is available."""
        return self.__has('PUN')

    def HasTGA(self):
        """Return True if TGA() is available."""
        return self.__has('TGA')

    def HasTGF(self):
        """Return True if TGF() is available."""
        return self.__has('TGF')

    def HasTGT(self):
        """Return True if TGT() is available."""
        return self.__has('TGT')

    def HasqTGT(self):
        """Return True if qTGT() is available."""
        return self.__has('qTGT')

    def HasqTWT(self):
        """Return True if qTWT() is available."""
        return self.__has('qTWT')

    def HasTGC(self):
        """Return True if TGC() is available."""
        return self.__has('TGC')

    def HasTGS(self):
        """Return True if TGS() is available."""
        return self.__has('TGS')

    def HasqTGL(self):
        """Return True if qTGL() is available."""
        return self.__has('qTGL')

    def HasATC(self):
        """Return True if ATC() is available."""
        return self.__has('ATC')

    def HasJOG(self):
        """Return True if JOG() is available."""
        return self.__has('JOG')

    def HasCMO(self):
        """Return True if CMO() is available."""
        return self.__has('CMO')

    def HasMVT(self):
        """Return True if MVT() is available."""
        return self.__has('MVT')

    def HasSTE(self):
        """Return True if STE() is available."""
        return self.__has('STE')

    def HasIMP(self):
        """Return True if IMP() is available."""
        return self.__has('IMP')

    def HasAAP(self):
        """Return True if AAP() is available."""
        return self.__has('AAP')

    def HasFIO(self):
        """Return True if FIO() is available."""
        return self.__has('FIO')

    def HasFSA(self):
        """Return True if FSA() is available."""
        return self.__has('FSA')

    def HasFAA(self):
        """Return True if FAA() is available."""
        return self.__has('FAA')

    def HasFAM(self):
        """Return True if FAM() is available."""
        return self.__has('FAM')

    def HasWFR(self):
        """Return True if WFR() is available."""
        return self.__has('WFR')

    def HasFAS(self):
        """Return True if FAS() is available."""
        return self.__has('FAS')

    def HasFSC(self):
        """Return True if FSC() is available."""
        return self.__has('FSC')

    def HasFSM(self):
        """Return True if FSM() is available."""
        return self.__has('FSM')

    def HasWAV_LIN(self):
        """Return True if WAV_LIN() is available."""
        return self.__has('WAV_LIN')

    def HasWAV_SWEEP(self):
        """Return True if WAV_SWEEP() is available."""
        return self.__has('WAV_SWEEP')

    def HasWAV_POL(self):
        """Return True if WAV_POL() is available."""
        return self.__has('WAV_POL')

    def HasWAV_SIN(self):
        """Return True if WAV_SIN() is available."""
        return self.__has('WAV_SIN')

    def HasWAV_TAN(self):
        """Return True if WAV_TAN() is available."""
        return self.__has('WAV_TAN')

    def HasWAV_RAMP(self):
        """Return True if WAV_RAMP() is available."""
        return self.__has('WAV_RAMP')

    def HasWAV_SIN_P(self):
        """Return True if WAV_SIN_P() is available."""
        return self.__has('WAV_SIN_P')

    def HasWAV_PNT(self):
        """Return True if WAV_PNT() is available."""
        return self.__has('WAV_PNT')

    def HasCTO(self):
        """Return True if CTO() is available."""
        return self.__has('CTO')

    def HasDRC(self):
        """Return True if DRC() is available."""
        return self.__has('DRC')

    def HasqDRC(self):
        """Return True if qDRC() is available."""
        return self.__has('qDRC')

    def HasSGA(self):
        """Return True if SGA() is available."""
        return self.__has('SGA')

    def HasNAV(self):
        """Return True if NAV() is available."""
        return self.__has('NAV')

    def HasDTC(self):
        """Return True if DTC() is available."""
        return self.__has('DTC')

    def HasEGE(self):
        """Return True if EGE() is available."""
        return self.__has('EGE')

    def HasMRT(self):
        """Return True if MRT() is available."""
        return self.__has('MRT')

    def HasMRW(self):
        """Return True if MRW() is available."""
        return self.__has('MRW')

    def HasMVE(self):
        """Return True if MVE() is available."""
        return self.__has('MVE')

    def HasMVR(self):
        """Return True if MVR() is available."""
        return self.__has('MVR')

    def HasNLM(self):
        """Return True if NLM() is available."""
        return self.__has('NLM')

    def HasOMA(self):
        """Return True if OMA() is available."""
        return self.__has('OMA')

    def HasOMR(self):
        """Return True if OMR() is available."""
        return self.__has('OMR')

    def HasPLM(self):
        """Return True if PLM() is available."""
        return self.__has('PLM')

    def HasRPA(self):
        """Return True if RPA() is available."""
        return self.__has('RPA')

    def HasSMO(self):
        """Return True if SMO() is available."""
        return self.__has('SMO')

    def HasSVA(self):
        """Return True if SVA() is available."""
        return self.__has('SVA')

    def HasSVR(self):
        """Return True if SVR() is available."""
        return self.__has('SVR')

    def HasVMA(self):
        """Return True if VMA() is available."""
        return self.__has('VMA')

    def HasVMI(self):
        """Return True if VMI() is available."""
        return self.__has('VMI')

    def HasqATZ(self):
        """Return True if qATZ() is available."""
        return self.__has('qATZ')

    def HasqBRA(self):
        """Return True if qBRA() is available."""
        return self.__has('qBRA')

    def HasqMVT(self):
        """Return True if qMVT() is available."""
        return self.__has('qMVT')

    def HasqSTE(self):
        """Return True if qSTE() is available."""
        return self.__has('qSTE')

    def HasqIMP(self):
        """Return True if qIMP() is available."""
        return self.__has('qIMP')

    def HasqCMO(self):
        """Return True if qCMO() is available."""
        return self.__has('qCMO')

    def HasqOMA(self):
        """Return True if qOMA() is available."""
        return self.__has('qOMA')

    def HasqCTV(self):
        """Return True if qCTV() is available."""
        return self.__has('qCTV')

    def HasqSMO(self):
        """Return True if qSMO() is available."""
        return self.__has('qSMO')

    def HasqSRA(self):
        """Return True if qSRA() is available."""
        return self.__has('qSRA')

    def HasAVG(self):
        """Return True if AVG() is available."""
        return self.__has('AVG')

    def HasJLT(self):
        """Return True if JLT() is available."""
        return self.__has('JLT')

    def HasCSV(self):
        """Return True if CSV() is available."""
        return self.__has('CSV')

    def HasBDR(self):
        """Return True if BDR() is available."""
        return self.__has('BDR')

    def HasCCL(self):
        """Return True if CCL() is available."""
        return self.__has('CCL')

    def HasCLR(self):
        """Return True if CLR() is available."""
        return self.__has('CLR')

    def HasCPY(self):
        """Return True if CPY() is available."""
        return self.__has('CPY')

    def HasMAC_DEF(self):
        """Return True if MAC_DEF() is available."""
        return self.__has('MAC_DEF')

    def HasREF(self):
        """Return True if REF() is available."""
        return self.__has('REF')

    def HasRBT(self):
        """Return True if RBT() is available."""
        return self.__has('RBT')

    def HasSAI(self):
        """Return True if SAI() is available."""
        return self.__has('SAI')

    def HasqAVG(self):
        """Return True if qAVG() is available."""
        return self.__has('qAVG')

    def HasqBDR(self):
        """Return True if qBDR() is available."""
        return self.__has('qBDR')

    def HasqRTR(self):
        """Return True if qRTR() is available."""
        return self.__has('qRTR')

    def HasqSCT(self):
        """Return True if qSCT() is available."""
        return self.__has('qSCT')

    def HasqSCH(self):
        """Return True if qSCH() is available."""
        return self.__has('qSCH')

    def HasqVST(self):
        """Return True if qVST() is available."""
        return self.__has('qVST')

    def HasqSSN(self):
        """Return True if qSSN() is available."""
        return self.__has('qSSN')

    def HasqHIS(self):
        """Return True if qHIS() is available."""
        return self.__has('qHIS')

    def HasqHDI(self):
        """Return True if qHDI() is available."""
        return self.__has('qHDI')

    def HasqHPV(self):
        """Return True if qHPV() is available."""
        return self.__has('qHPV')

    def HasqCCL(self):
        """Return True if qCCL() is available."""
        return self.__has('qCCL')

    def HasqCSV(self):
        """Return True if qCSV() is available."""
        return self.__has('qCSV')

    def HasqERR(self):
        """Return True if qERR() is available."""
        return self.__has('qERR')

    def HasqSWT(self):
        """Return True if qSWT() is available."""
        return self.__has('qSWT')

    def HasMAC_qFREE(self):
        """Return True if MAC_qFREE() is available."""
        return self.__has('MAC_qFREE')

    def HasqFSS(self):
        """Return True if qFSS() is available."""
        return self.__has('qFSS')

    def HasqGFL(self):
        """Return True if qGFL() is available."""
        return self.__has('qGFL')

    def HasqTAC(self):
        """Return True if qTAC() is available."""
        return self.__has('qTAC')

    def HasqTIM(self):
        """Return True if qTIM() is available."""
        return self.__has('qTIM')

    def HasqTIO(self):
        """Return True if qTIO() is available."""
        return self.__has('qTIO')

    def HasqWFR(self):
        """Return True if qWFR() is available."""
        return self.__has('qWFR')

    def HasqTLT(self):
        """Return True if qTLT() is available."""
        return self.__has('qTLT')

    def HasqTNR(self):
        """Return True if qTNR() is available."""
        return self.__has('qTNR')

    def HasqTNJ(self):
        """Return True if qTNJ() is available."""
        return self.__has('qTNJ')

    def HasqTPC(self):
        """Return True if qTPC() is available."""
        return self.__has('qTPC')

    def HasqTSC(self):
        """Return True if qTSC() is available."""
        return self.__has('qTSC')

    def HasqTWG(self):
        """Return True if qTWG() is available."""
        return self.__has('qTWG')

    def HasqVLS(self):
        """Return True if qVLS() is available."""
        return self.__has('qVLS')

    def HasqIDN(self):
        """Return True if qIDN() is available."""
        return self.__has('qIDN')

    def HasqVER(self):
        """Return True if qVER() is available."""
        return self.__has('qVER')

    def HasqTVI(self):
        """Return True if qTVI() is available."""
        return self.__has('qTVI')

    def HasqVAR(self):
        """Return True if qVAR() is available."""
        return self.__has('qVAR')

    def HasBRA(self):
        """Return True if BRA() is available."""
        return self.__has('BRA')

    def HasqCST(self):
        """Return True if qCST() is available."""
        return self.__has('qCST')

    def HasqPUN(self):
        """Return True if qPUN() is available."""
        return self.__has('qPUN')

    def HasqECO(self):
        """Return True if qECO() is available."""
        return self.__has('qECO')

    def HasqKEN(self):
        """Return True if qKEN() is available."""
        return self.__has('qKEN')

    def HasqKLN(self):
        """Return True if qKLN() is available."""
        return self.__has('qKLN')

    def HasqKET(self):
        """Return True if qKET() is available."""
        return self.__has('qKET')

    def HasqDCO(self):
        """Return True if qDCO() is available."""
        return self.__has('qDCO')

    def HasqEAX(self):
        """Return True if qEAX() is available."""
        return self.__has('qEAX')

    def HasqLIM(self):
        """Return True if qLIM() is available."""
        return self.__has('qLIM')

    def HasONL(self):
        """Return True if ONL() is available."""
        return self.__has('ONL')

    def HasWCL(self):
        """Return True if WCL() is available."""
        return self.__has('WCL')

    def HasqDRL(self):
        """Return True if qDRL() is available."""
        return self.__has('qDRL')

    def HasqWGC(self):
        """Return True if qWGC() is available."""
        return self.__has('qWGC')

    def HasqWGO(self):
        """Return True if qWGO() is available."""
        return self.__has('qWGO')

    def HasqWMS(self):
        """Return True if qWMS() is available."""
        return self.__has('qWMS')

    def HasqWTR(self):
        """Return True if qWTR() is available."""
        return self.__has('qWTR')

    def HasqCTI(self):
        """Return True if qCTI() is available."""
        return self.__has('qCTI')

    def HasqCTO(self):
        """Return True if qCTO() is available."""
        return self.__has('qCTO')

    def HasqDIA(self):
        """Return True if qDIA() is available."""
        return self.__has('qDIA')

    def HasqDRT(self):
        """Return True if qDRT() is available."""
        return self.__has('qDRT')

    def HasqHDT(self):
        """Return True if qHDT() is available."""
        return self.__has('qHDT')

    def HasqFSN(self):
        """Return True if qFSN() is available."""
        return self.__has('qFSN')

    def HasqFED(self):
        """Return True if qFED() is available."""
        return self.__has('qFED')

    def HasqHIA(self):
        """Return True if qHIA() is available."""
        return self.__has('qHIA')

    def HasqTWE(self):
        """Return True if qTWE() is available."""
        return self.__has('qTWE')

    def HasqHIB(self):
        """Return True if qHIB() is available."""
        return self.__has('qHIB')

    def HasqSRG(self):
        """Return True if qSRG() is available."""
        return self.__has('qSRG')

    def HasqSTA(self):
        """Return True if qSTA() is available."""
        return self.__has('qSTA')

    def HasGetStatus(self):
        """Return True if GetStatus() is available."""
        return self.__has('GetStatus')

    def HasqWAV(self):
        """Return True if qWAV() is available."""
        return self.__has('qWAV')

    def HasqTRA(self):
        """Return True if qTRA() is available."""
        return self.__has('qTRA')

    def HasqKLC(self):
        """Return True if qKLC() is available."""
        return self.__has('qKLC')

    def HasqKLS(self):
        """Return True if qKLS() is available."""
        return self.__has('qKLS')

    def HasqKLT(self):
        """Return True if qKLT() is available."""
        return self.__has('qKLT')

    def HasqWGS(self):
        """Return True if qWGS() is available."""
        return self.__has('qWGS')

    def HasqMAS(self):
        """Return True if qMAS() is available."""
        return self.__has('qMAS')

    def HasqHIE(self):
        """Return True if qHIE() is available."""
        return self.__has('qHIE')

    def HasqHIL(self):
        """Return True if qHIL() is available."""
        return self.__has('qHIL')

    def HasqJAS(self):
        """Return True if qJAS() is available."""
        return self.__has('qJAS')

    def HasqJAX(self):
        """Return True if qJAX() is available."""
        return self.__has('qJAX')

    def HasqMOD(self):
        """Return True if qMOD() is available."""
        return self.__has('qMOD')

    def HasqJBS(self):
        """Return True if qJBS() is available."""
        return self.__has('qJBS')

    def HasqVMO(self):
        """Return True if qVMO() is available."""
        return self.__has('qVMO')

    def HasqWGI(self):
        """Return True if qWGI() is available."""
        return self.__has('qWGI')

    def HasqWGN(self):
        """Return True if qWGN() is available."""
        return self.__has('qWGN')

    def HasqWSL(self):
        """Return True if qWSL() is available."""
        return self.__has('qWSL')

    def HasqDTL(self):
        """Return True if qDTL() is available."""
        return self.__has('qDTL')

    def HasqONL(self):
        """Return True if qONL() is available."""
        return self.__has('qONL')

    def HasqOSN(self):
        """Return True if qOSN() is available."""
        return self.__has('qOSN')

    def HasqTRO(self):
        """Return True if qTRO() is available."""
        return self.__has('qTRO')

    def HasqTRI(self):
        """Return True if qTRI() is available."""
        return self.__has('qTRI')

    def HasqJON(self):
        """Return True if qJON() is available."""
        return self.__has('qJON')

    def HasqDIP(self):
        """Return True if qDIP() is available."""
        return self.__has('qDIP')

    def HasqEGE(self):
        """Return True if qEGE() is available."""
        return self.__has('qEGE')

    def HasqFES(self):
        """Return True if qFES() is available."""
        return self.__has('qFES')

    def HasqFRF(self):
        """Return True if qFRF() is available."""
        return self.__has('qFRF')

    def HasqHAR(self):
        """Return True if qHAR() is available."""
        return self.__has('qHAR')

    def HasqHIN(self):
        """Return True if qHIN() is available."""
        return self.__has('qHIN')

    def HasqIFC(self):
        """Return True if qIFC() is available."""
        return self.__has('qIFC')

    def HasqIFS(self):
        """Return True if qIFS() is available."""
        return self.__has('qIFS')

    def HasqONT(self):
        """Return True if qONT() is available."""
        return self.__has('qONT')

    def HasqOVF(self):
        """Return True if qOVF() is available."""
        return self.__has('qOVF')

    def HasqMAN(self):
        """Return True if qMAN() is available."""
        return self.__has('qMAN')

    def HasqMAC(self):
        """Return True if qMAC() is available."""
        return self.__has('qMAC')

    def HasqHPA(self):
        """Return True if qHPA() is available."""
        return self.__has('qHPA')

    def HasqHDR(self):
        """Return True if qHDR() is available."""
        return self.__has('qHDR')

    def HasqRMC(self):
        """Return True if qRMC() is available."""
        return self.__has('qRMC')

    def HasqREF(self):
        """Return True if qREF() is available."""
        return self.__has('qREF')

    def HasqRON(self):
        """Return True if qRON() is available."""
        return self.__has('qRON')

    def HasqRTO(self):
        """Return True if qRTO() is available."""
        return self.__has('qRTO')

    def HasqSAI_ALL(self):
        """Return True if qSAI_ALL() is available."""
        return self.__has('qSAI_ALL')

    def HasqSSL(self):
        """Return True if qSSL() is available."""
        return self.__has('qSSL')

    def HasqTRS(self):
        """Return True if qTRS() is available."""
        return self.__has('qTRS')

    def HasqVCO(self):
        """Return True if qVCO() is available."""
        return self.__has('qVCO')

    def HasDIO(self):
        """Return True if DIO() is available."""
        return self.__has('DIO')

    def HasqDIO(self):
        """Return True if qDIO() is available."""
        return self.__has('qDIO')

    def HasSGP(self):
        """Return True if SGP() is available."""
        return self.__has('SGP')

    def HasqSGP(self):
        """Return True if qSGP() is available."""
        return self.__has('qSGP')

    def HasSPA(self):
        """Return True if SPA() is available."""
        return self.__has('SPA')

    def HasSEP(self):
        """Return True if SEP() is available."""
        return self.__has('SEP')

    def HasqSPA(self):
        """Return True if qSPA() is available."""
        return self.__has('qSPA')

    def HasqSEP(self):
        """Return True if qSEP() is available."""
        return self.__has('qSEP')

    def HasqLST(self):
        """Return True if qLST() is available."""
        return self.__has('qLST')

    def HasPOL(self):
        """Return True if POL() is available."""
        return self.__has('POL')

    def HasSTD(self):
        """Return True if STD() is available."""
        return self.__has('STD')

    def HasRTD(self):
        """Return True if RTD() is available."""
        return self.__has('RTD')

    def HasqRTD(self):
        """Return True if qRTD() is available."""
        return self.__has('qRTD')

    def HasDLT(self):
        """Return True if DLT() is available."""
        return self.__has('DLT')
