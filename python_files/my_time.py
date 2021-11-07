# usage:
# import my_time as mt
# t = mt.my_time()
# t.tic()
# function()
# t.toc()

import time

class my_time:
    def __init__(self):
        self.t0 = 0
        self.t1 = 0

    def tic(self):
        self.t0 = time.time()

    def toc(self):
        self.t1 = time.time()
        print("Elapsed time: %f seconds.\n" %(self.t1-self.t0))
        return self.t1-self.t0
