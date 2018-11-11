import queue
import threading
import multiprocessing
from .datasamplers import RandomDataSampler
from .datasamplers import BatchDataSampler
class DataLoader(object):
    """Data batch loader.
    Attributes:
        dataset (Dataset): A dataset
        batch_size (int, optional): The batch size, defaults to 1
        epochs (int, optional): The number of epochs, defaults to 1
        drop_last (bool, optional): Drop last batch if it's smaller than the batch size, defaults to True
        data_sampler (DataSampler, optional): Data sampler, defaults to RandomDataSampler
        num_workers (int, optional): The number of background threads loading data onto queue, defaults to 1
        batch_sampler(BatchDataSampler): Sampler for batching according to the `data_sampler` scheme.
        idx_queue (Queue): Queue holding batch indices for data loading
        data_queue (Queue): Queue holding batches of data for data loading.
    """
    def __init__(self, dataset, batch_size=1, epochs=1, drop_last=True, data_sampler=None, num_workers=1,
                 use_multiprocessing=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.epochs = epochs
        self.num_workers = num_workers
        self.workers = []
        self.data_sampler = data_sampler if data_sampler else RandomDataSampler
        self.batch_sampler = BatchDataSampler(dataset=self.dataset, data_sampler=data_sampler,
                                              batch_size=batch_size, drop_last=drop_last)
        self.idx_queue = None
        self.data_queue = None
        self.stop_event = None
        self.stop_flag = False
        self.use_multiprocessing = use_multiprocessing
    def request_start(self):
        # Initialize data readers for the dataset
        self.dataset.init_readers()
        # Initialize queues
        self.idx_queue = multiprocessing.Queue() if self.use_multiprocessing else queue.Queue()
        self.data_queue = multiprocessing.Queue() if self.use_multiprocessing else queue.Queue()
        self.stop_event = multiprocessing.Event() if self.use_multiprocessing else threading.Event()
        # Add batch idx samples
        for _ in range(self.epochs):
            samples = self.batch_sampler.sample()
            for s in samples:
                self.idx_queue.put(s)
        # Create workers
        args = (self.idx_queue, self.data_queue, self.dataset, self.stop_event)
        for _ in range(self.num_workers):
            self.workers.append(multiprocessing.Process(target=do_work, args=args) if self.use_multiprocessing
                                else threading.Thread(target=do_work, args=args))
        # Starts workers
        for worker in self.workers:
            worker.daemon = True
            worker.start()
    def request_stop(self):
        """Stops data loading threads."""
        self.stop_event.set()
        self.stop_flag = True
        self._clear_queues()
    def should_stop(self):
        """Checks if data loader should stop."""
        return self.stop_flag or (self.idx_queue.empty() and self.data_queue.empty())
    def next(self):
        """Gets the next batch of data from the data queue."""
        return None if self.stop_flag else self.data_queue.get()
    def _clear_queues(self):
        """Clear all enqueued content items."""
        while not self.idx_queue.empty():
            self.idx_queue.get()
        while not self.data_queue.empty():
            self.data_queue.get()
def do_work(idx_queue, data_queue, dataset, stop_event):
    """Reads batches of indices and puts the corresponding data points onto the data queue."""
    while not (stop_event.is_set() or idx_queue.empty()):
        idxs = idx_queue.get()
        data_queue.put(dataset.get(idxs, 0))