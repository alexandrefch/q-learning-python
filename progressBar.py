import sys

def progressbar(it, prefix="", size=60, file=sys.stdout):
    """
    Print a progress bar

    """
    count = len(it)
    def show(j):
        x = int(size*j/count)
        file.write("%s [%s%s] %.0f %% \r" % (prefix.ljust(12), "═"*x, "."*(size-x), j*100/count))
        file.flush()        
    show(0)
    for i, item in enumerate(it):
        yield item
        show(i+1)
    file.write("\n")
    file.flush()
