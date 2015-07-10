import multiprocessing

def worker(num):
    """thread worker function"""
    print 'Worker:', num
    return

if __name__ == '__main__':
    jobs = []
    print 'start?'
    for i in range(100):
        p = multiprocessing.Process(target=worker, args=(i,))
        jobs.append(p)
        p.start()
