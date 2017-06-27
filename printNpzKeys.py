#!/usr/bin/env python

# utility to print the keys of an npz file

import sys
import numpy as np
from pprint import pprint

ARGV = sys.argv[1:]

assert len(ARGV) == 1

fname = ARGV.pop(0)

data = np.load(fname)

pprint(sorted(data.keys()))



