#!/usr/bin/env python

# do NOT run setup.sh, we were unable to build libxml2 for our python version

import sys
import libxml2

#----------------------------------------------------------------------
# main
#----------------------------------------------------------------------

ARGV = sys.argv[1:]

assert len(ARGV) == 1

printOnlyModified = False

#----------

doc = libxml2.parseFile(ARGV[0])
ctxt = doc.xpathNewContext()
options = ctxt.xpathEval("//MethodSetup/Options/Option")

#----------

print '":".join(['

for opt in options:
    
    paramName = opt.prop("name")

    modified = opt.prop("modified") != 'No'

    if printOnlyModified and not modified:
        continue

    paramValue = opt.get_content()

    if paramValue.lower() == 'false':
        print '  "!%s",' % paramName
    elif paramValue.lower() == 'true':
        print '  "%s",' % paramName
    else:
        print '  "%s=%s",' % (paramName, paramValue)


print '])'
