import os, torch, torch.distributed as dist

def main():
    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl")

    x = torch.ones(1, device=rank)
    dist.all_reduce(x)
    print(f"[rank {rank}] OK: {x}")

if __name__ == "__main__":
    main()
