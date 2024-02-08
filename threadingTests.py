from multiprocessing import Process
import tools

def print_func(continent='Asia'):
    print('The name of continent is : ', continent)


def print_test(continent='Asia'):
    print('OTHER PRINT The name of continent is : ', continent)


if __name__ == "__main__":  # confirms that the code is under main function
    names = ['America', 'Europe', 'Africa']
    procs = []
    proc = Process(target=tools.get_phrase())  # instantiating without any argument
    procs.append(proc)
    proc.start()

    # instantiating process with arguments
    for name in names:
        # print(name)
        proc = Process(target=print_func, args=(name,))
        procs.append(proc)
        proc.start()

    # complete the processes
    for proc in procs:
        proc.join()