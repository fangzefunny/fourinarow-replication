import numpy as np
from multiprocessing import Pool, Manager
from functools import partial
from copy import deepcopy
import time

def evaluate_point(params, shared_dict, dataset_index):
    """Worker function that evaluates one data point with given parameters"""
    # Get dataset from shared dictionary
    x, y_true = shared_dict['dataset'][dataset_index]
    
    # Compute predicted y using our model: y = a*x^2 + b*x + c
    a, b, c = params
    y_pred = a*(x**2) + b*x + c
    
    # Calculate error for this point
    error = (y_pred - y_true)**2
    
    # Update shared dictionary with error
    shared_dict['errors'][dataset_index] = error
    
    # Update shared counter 
    shared_dict['counter'].value += 1
    
    return error

def worker(T, k):
    time.sleep(.005)
    success = k==(T-1)
    return success

def process_worker(shared_tasks, shared_lock, worker_id):
    print(f'worker {worker_id} started')

    while True:
        # get all unfinished tasks
        print(f'worker {worker_id} getting unfinished tasks')
        unfinished_ids = []
        if worker_id == 0: time.sleep(5)
        for task_id, (_, task) in enumerate(shared_tasks.items()):
            if not task['success']: 
                unfinished_ids.append(task_id)
        if len(unfinished_ids) == 0: break 
        print(f'worker {worker_id} found {len(unfinished_ids)} unfinished tasks')

        # sample one unfinished item
        selected_idx = np.random.choice(unfinished_ids)
        unfinished_task = deepcopy(shared_tasks[selected_idx])
        print(f'worker {worker_id}: {unfinished_ids}')
        print(f'worker {worker_id}: {unfinished_task}')
        
        # Keep processing until task is complete
        while not shared_tasks[selected_idx]['success']:
            with shared_lock:
                # Get latest values
                k = unfinished_task['k']
                success = worker(unfinished_task['state'], k)
                
                if success:
                    unfinished_task['success'] = True
                    unfinished_task['k'] += 1
                    shared_tasks[selected_idx] = unfinished_task
                else:
                    unfinished_task['k'] += 1
                    shared_tasks[selected_idx] = unfinished_task

def parallel_evaluate(dataset, n_workers=2):
    """Evaluate parameters against dataset in parallel"""
    # Initialize shared data using Manager
    N = len(dataset)
    manager = Manager()
    shared_tasks = manager.dict(dataset)
    shared_lock = manager.Lock()

    pool = Pool(n_workers)
    # Create pool and run evaluations in parallel
    results = [pool.apply_async(process_worker, 
                                args=(shared_tasks, shared_lock, i)) 
                                for i in range(n_workers)]
    [result.get() for result in results]

    # calculate the counter
    K = [task['k'] for task in shared_tasks.values()]
    return K
    
def loop_evaluate(trial = [1, 2, 4, 6, 8, 10, 100, 500]):
    N = len(trial)
    K = np.zeros([N])

    for i in range(N):
        k, success = 0, False
        while not success:
            success = worker(trial[i], k)
            k += 1
        K[i] = k

    return K

def compare_evaluations(trials = [1, 2, 4, 6, 8, 10, 100, 500]):
    """Find parameters that best fit our quadratic model"""
   
    # loop_evaluate
    print('loop_evaluate')
    np.random.seed(123)
    start = time.perf_counter()
    K1 = loop_evaluate(trials)
    print(f"Time taken: {time.perf_counter() - start:.4f} seconds")
    print(K1)

    print('parallel_evaluate')
    np.random.seed(123)
    start = time.perf_counter()
    dataset = {i:{
        'state': trials[i],
        'success': False,
        'k': 0,
    } for i in range(len(trials))}
    # Evaluate each parameter set in parallel
    K2 = parallel_evaluate(dataset, n_workers=10)
    print(f"Time taken: {time.perf_counter() - start:.4f} seconds")
    print(K2)

if __name__ == "__main__":
    compare_evaluations([1, 2, 4, 6, 8, 10, 100, 500])