import multiprocessing as mp

def worker(_):
    import jax
    print("worker jax ok")

if __name__ == "__main__":
    ctx = mp.get_context("spawn")
    pool = ctx.Pool(2)
    pool.map(worker, [0, 1])
