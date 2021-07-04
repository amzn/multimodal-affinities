import traceback
import logging

try:
    import multiprocessing
except ImportError:
    multiprocessing = None


class MultiprocessHelper:
    """
    An small framework for running heavy operations as multi-processes.
    MultiprocessHelper uses a map-reduce architecture, where multiple read workers perform the mapping
    and a single write worker performs reduce.
    The final results processed results are returned as a list of results (may be empty).
    """

    def __init__(self, num_processes=multiprocessing.cpu_count(), queue_size=1024):
        """
        Initialize the helper and ready it for a multi-process batch job.
        :param num_processes: Number of processes to use. Default: cpu count.
        Note: The default value may depend on operating systems definitions and may be inaccurate.
        :param queue_size: The buffer cache size for each worker thread. This is the max number of entries from
        the workload that each worker may hold at any present time.
        Normally you shouldalter this value only to prevent out of memory errors.
        """
        self.num_processes = num_processes
        self.queue_size = queue_size
        read_workers_count = max(1, num_processes-1)
        self.read_queues = [multiprocessing.Queue(queue_size) for _ in range(read_workers_count)]
        self.write_queue = multiprocessing.Queue(queue_size)
        self.results_queue = multiprocessing.Queue(queue_size)
        logging.debug('MultiprocessHelper :: Initialized Multiprocess Helper with %r processes and queue_size=%r. '
                      'Main process id=%r' % (num_processes, queue_size, self.worker_id()))

    @staticmethod
    def worker_id():
        """
        :return: Unique id of the current worker process.
        """
        return multiprocessing.current_process().pid

    def execute(self, workload, read_func, write_func=None, read_args=(), write_args=(),
                aggregation_type=list):
        """
        Executes the map-reduce operation on the workload, using multiple processes.

                              [map]            [reduce]

        ============   ->   read_func                            ============
        | Workload |    ->   read_func    ->  write_func   ->    | Results  |
        ============   ->   read_func                            ============

        :param workload: An iterable object containing the data entries.
        :param read_func: A "map function", which accepts a single data entry and performs some
        mapping / transformation operation on it.
        The exact signature of read_func is func(entry_idx, data_entry, *args), where
        entry_idx is a unique running index of the current entry from the workload,
        data_entry is the content of the current entry from the workload, and args is a tuple
        containing any number of additional optional arguments needed by this function (may be empty).
        Note: the return value of this function may be limited by the type of objects possible to share
        between processes. In general, those should be simple, canonical objects (i.e: str).
        For example: objects containing c-pointers cannot be returned by this function.
        :param write_func: A "reduce function", which accepts a single data entry and performs some
        aggregation operation on it.
        The exact signature of write_func is func(entry_idx, data_entry, aggregation, *args), where
        entry_idx is a unique running index of the current entry from the workload,
        data_entry is the content of the processed entry as outputted from the mapping function,
        aggregation is a reusable cache (list) the function may utilize to keep information between
        repeated invocations, and args is a tuple containing any number of additional optional arguments
        needed by this function (may be empty).
        Note: the return value of this function may be limited by the type of objects possible to share
        between processes. In general, those should be simple, canonical objects (i.e: str).
        For example: objects containing c-pointers cannot be returned by this function.
        :param read_args: Tuple. Optional arguments needed by read_func.
        :param write_args: Tuple. Optional arguments needed by write_func.
        :param aggregation_type: Type of iterable container to contain aggregated results, as used
        by the write worker.
        :return: A list containing the results of write_func.
        """
        # Initialize an inner object that helps balance the workload between multiple workers
        workload_alloc = WorkloadAllocator(workload)

        # Create the reader / writer workers
        read_processes, write_process = self.prepare_processes(read_func, write_func,
                                                               read_args, write_args,
                                                               aggregation_type)

        # Start the workers. They will block until data from the workload is fed.
        self.start_processes(read_processes, write_process)

        # Feed the initial batch of data to the reader workers.
        workload_alloc.divide_workload(self.read_queues)

        # Wait for the reader / writer workers to finish processing
        results = self.async_process(read_processes, write_process)
        logging.debug('MultiprocessHelper :: Execution finished')

        return results

    def prepare_processes(self, read_func, write_func, read_args, write_args, aggregation_type):
        """
        Creates the reader / writer workers.
        Note: The workers are not started until start_processes is called.
        :param read_func: A map function to be used by the reader workers.
        :param write_func: A reduce function to be used the writer worker.
        :param read_args: Tuple. Optional arguments needed by read_func.
        :param write_args: Tuple. Optional arguments needed by write_func.
        :param aggregation_type: Type of iterable container to contain aggregated results, as used
        by the write worker.
        :return: The initialized read / write workers
        """
        read_processes = []

        for idx in range(len(self.read_queues)):
            read_worker_params = (read_func, read_args,
                                  self.read_queues[idx], self.write_queue)
            read_processes.append(multiprocessing.Process(target=self.read_worker, args=read_worker_params))
        write_process = multiprocessing.Process(target=self.write_worker,
                                                args=(write_func, write_args,
                                                      self.write_queue, self.results_queue, aggregation_type))
        return read_processes, write_process

    @staticmethod
    def start_processes(read_processes, write_process):
        """
        Starts the reader and writer workers.
        :param read_processes: List of multiple reader workers running the map function.
        :param write_process: Single writer worker running the reduce function.
        """
        for process in read_processes:
            process.start()

        write_process.start()

    def async_process(self, read_processes, write_process):
        """
        Blocks until all reader & writer workers finish processing.
        Note: This function doesn't use a timeout, to support very long operations.
        :param read_processes: List of multiple reader workers running the map function.
        :param write_process: Single writer worker running the reduce function.
        :return: A list of results (or single item), as returned by the write worker.
        """
        for process in read_processes:
            while process.is_alive():
                process.join(timeout=1)
                logging.debug('MultiprocessHelper :: Main process - waiting for reader #%r' % process.pid)
            logging.debug('MultiprocessHelper :: Main process - Read Worker collected')

        self.write_queue.put(None)
        results = self.collect_results(self.results_queue)  # Wait for writer to finish and put load on queue
        while write_process.is_alive():
            write_process.join(timeout=1)
            logging.debug('MultiprocessHelper :: Main process - waiting for writer #%r' % process.pid)
        logging.debug('MultiprocessHelper :: Main process - Write Worker collected')
        return results

    @staticmethod
    def read_worker(read_func, read_func_args, read_queue, write_queue):
        """
        Read worker main function.
        Consumes workload data allocated from workload_alloc, and runs read_func(), the map function,
        on each entry.
        The transformed results are placed inside the writer queue.
        :param read_func: A map function to be used by the reader workers.
        :param read_func_args: Tuple. Optional arguments needed by read_func.
        :param read_queue: A queue for caching current workloaded this worker have allocated from workload_alloc.
        By the beginning of this function, read_queue is already filled with some data.
        :param write_queue: A queue for storing the processed entries of this worker.
        """
        logging.debug('MultiprocessHelper :: Process [Read Worker] #%r started..' % MultiprocessHelper.worker_id())
        while True:
            # Next entry
            chunk = read_queue.get()
            if chunk is None:   # Finished current allocation.
                logging.debug('MultiprocessHelper :: Process [Read Worker] #%r has no more workload..' %
                              MultiprocessHelper.worker_id())
                break   # Workload have been processed entirely..

            entry_idx, data_entry = chunk
            try:
                # Run the map function
                processed_entry = read_func(entry_idx, data_entry, *read_func_args)
                write_queue.put((entry_idx, processed_entry))
                logging.debug('MultiprocessHelper :: Process [Read Worker] #%r finished processing entry %r..' %
                              (MultiprocessHelper.worker_id(), entry_idx))
            except Exception:
                logging.error('MultiprocessHelper :: Process [Read Worker] #%r have '
                              'run into error during read_func()..' % MultiprocessHelper.worker_id())
                traceback.print_exc()
                write_queue.put((entry_idx, None))
        logging.debug('MultiprocessHelper :: Process [Read Worker] #%r exiting..' % MultiprocessHelper.worker_id())

    @staticmethod
    def write_worker(write_func, write_func_args, write_queue, results_queue, aggregation_type):
        """
        Write worker main function.
        Consumes workload data aggregated from all reader workers, and runs write_func(), the reduce function,
        on each entry.
        The aggregated results are placed inside the results queue.
        :param write_func: A reduce function to be used by the write worker.
        :param write_func_args: Tuple. Optional arguments needed by write_func.
        :param write_queue: A queue for caching current workloaded this worker have accumulated from the reader
        workers.
        :param results_queue: A queue for storing the processed entries of this worker to the main process.
        :param aggregation_type: Type of iterable container to contain aggregated results, as used
        by the write worker.
        """
        logging.debug('MultiprocessHelper :: Process [Write Worker] #%r started..' % MultiprocessHelper.worker_id())
        results = None
        aggregation = aggregation_type()
        while True:
            chunk = write_queue.get()
            if chunk is None:
                logging.debug('MultiprocessHelper :: Process [Write Worker] #%r has no more workload..' %
                              MultiprocessHelper.worker_id())
                break
            entry_idx, processed_entry = chunk
            if processed_entry is not None:
                if write_func is not None:
                    results = write_func(entry_idx, processed_entry, aggregation, *write_func_args)
                    logging.debug('MultiprocessHelper :: Process [Write Worker] #%r finished processing %r..' %
                                  (MultiprocessHelper.worker_id(), entry_idx))
            else:
                logging.error('MultiprocessHelper :: Process [Write Worker] #%r have encountered a'
                              ' bad entry: index=%r' % (MultiprocessHelper.worker_id(), entry_idx))
        logging.debug('MultiprocessHelper :: Process [Write Worker] #%r exiting..' % MultiprocessHelper.worker_id())

        # The last call to write_func determines the results
        if results is not None:
            results_queue.put(results)
        results_queue.put(None)

    @staticmethod
    def collect_results(results_queue):
        """
        Consumes the results queue containing the write worker aggregation results, and returns it as a list.
        :param results_queue:
        :return: A list of results (or single item), as returned by the write worker.
        """
        collected_results = []
        while True:
            next_result = results_queue.get()
            if next_result is None:
                break
            collected_results.append(next_result)
        if collected_results is None:
            return None
        return collected_results[0] if len(collected_results) == 1 else collected_results


class WorkloadAllocator:
    """
    A helper object for allocating data from the workload to the worker processes.
    """

    def __init__(self, workload):
        """
        :param workload: Iterable containing the data.
        :param queue_size: Cache size of the workers, used to calculate the initial capacity.
        """
        self.next_workload_entry_idx = 0
        self.workload_iterator = iter(workload)

    def divide_workload(self, read_queues):
        """
        Divides workload among the read workers.
        :param read_queues: List of queues to cache data from the workload
        :return: read_queues are filled with data from the workload, and finally with None to symbolize the
        end of the queue (prevent blocking)
        """
        logging.debug('MultiprocessHelper :: Dividing workload to %r read queues' %
                      len(read_queues))

        while True:
            try:
                item = next(self.workload_iterator)
            except StopIteration:
                logging.debug('MultiprocessHelper :: Iteration finished..')
                break
            # Divide data in round robin fashion
            next_queue = self.next_workload_entry_idx % len(read_queues)
            data_entry = (self.next_workload_entry_idx, item)
            read_queues[next_queue].put(data_entry)
            self.next_workload_entry_idx += 1

        logging.debug('MultiprocessHelper :: Workload exhausted, sealing queue ends..')
        # Mark the end of queue to prevent reader workers from blocking
        for queue in read_queues:
            queue.put(None)
        logging.debug('MultiprocessHelper :: Workload allocator quitting')
