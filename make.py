#!/usr/bin/env python

import sys
import os

def mkdir(dir):
    if not os.access(dir, os.F_OK):
        os.mkdir(dir)

def run(cmd):
    import subprocess
    print cmd
    proc = subprocess.Popen(cmd, shell=True, stdout=sys.stdout, stderr=sys.stderr, stdin=sys.stdin)
    returncode = proc.wait()
    if 0 != returncode:
        raise Exception("'%s' failed %d" % (cmd, returncode))

def getCPUCount():
    try:
        import multiprocessing
        return multiprocessing.cpu_count()
    except (ImportError,NotImplementedError):
        print >>sys.stderr, "warn: can't determine cpu count"
        pass
    return 8

APP_NAME = "kaggle-solutions"
ROOT_FOLDER = "/%s" % APP_NAME

def splitCwd(cwd, buildDir):
    i = cwd.find(ROOT_FOLDER)
    return cwd[:i + 1] + buildDir, cwd[i + len(ROOT_FOLDER) + 1:]

def handleOptions():
    from optparse import OptionParser
    parser = OptionParser()
    parser.add_option("-d", "--debug", action="store_true", dest="debug", help="make debug", default=False)
    parser.add_option("-e", "--eclipse", action="store_true", dest="eclipse", help="eclipse", default=False)
    (options, args) = parser.parse_args()
    return options

def make(path, threads):
    cmd = "cd %s; make -j%d" % (path, threads)
    run(cmd)

def mainCMake():
    cwd = os.getcwd()

    options = handleOptions()
    if options.eclipse:
        mode = "Eclipse"
    elif options.debug:
        mode = "Debug"
    else:
        mode = "Release"
    
    buildHome, projectParts = splitCwd(os.getcwd(), "%s-%s" % (APP_NAME, mode.lower()))
    mkdir(buildHome)
    print >> sys.stderr, "buildHome=%s\nprojectParts=%s\n" % (buildHome, projectParts)
    
    makeOnly = ""
    gen = ""
    if options.eclipse:
        sMode = "Debug"
        gen = "-G \"Eclipse CDT4 - Unix Makefiles\""
    run("cd %s; cmake -DCMAKE_BUILD_TYPE=%s %s ../%s" % (buildHome, mode, gen, APP_NAME))

    threadCount = getCPUCount()
    if not options.eclipse:
        make(buildHome, threadCount)
    else:
        run("~/eclipse/eclipse %s" % buildHome)
    
    run("rm -f %s/digits" % (cwd))
    run("ln -s %s/digits %s" % (buildHome, cwd))

if __name__ == "__main__":
    mainCMake()
