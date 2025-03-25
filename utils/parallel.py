import multiprocessing as mp

# --------- Get pools  --------- #  

def get_pool(args):
    n_cores = args.n_cores if args.n_cores else int(mp.cpu_count()*.7) 
    print(f'    Using {n_cores} parallel CPU cores\n ')
    return mp.Pool(processes=n_cores)