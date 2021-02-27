import os, threading


def get_ident(thread_safe=True):
    global ident
    thread = 'default'
    if not thread_safe:
        thread = threading.get_ident()
    return "{}#{}".format(os.getpid(), thread)