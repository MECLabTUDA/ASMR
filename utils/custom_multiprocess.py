'''
Custom non-daemonic Pool class
Code adapted from https://stackoverflow.com/questions/6974695/python-process-pool-non-daemonic
'''
import multiprocessing
import multiprocessing.pool

'''
class NoDaemonProcess(multiprocessing.Process):
    @property
    def _get_daemon(self):
        return False

    @daemon.setter
    def _set_daemon(self, value):
        pass
    #daemon = property(_get_daemon, _set_daemon)

#class MyPool(multiprocessing.pool.Pool):
class MyPool(type(multiprocessing.get_context())):
    Process = NoDaemonProcess

'''

class NoDaemonProcess(multiprocessing.Process):
    @property
    def daemon(self):
        return False

    @daemon.setter
    def daemon(self, value):
        pass


class MyPool(multiprocessing.pool.Pool):
   def Process(self, *args, **kwds):
        proc = super(MyPool, self).Process(*args, **kwds)
        proc.__class__ = NoDaemonProcess

        return proc
