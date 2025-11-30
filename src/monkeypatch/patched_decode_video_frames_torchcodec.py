import lerobot.common.datasets.video_utils as video_utils

def decode_video_frames_torchcodec_patched(
    video_path,
    timestamps,
    tolerance_s,
    device="cpu",
    log_loaded_timestamps=False,
):
    from torchcodec.decoders import VideoDecoder

    decoder = VideoDecoder(video_path, device=device, seek_mode="approximate")
    metadata = decoder.metadata
    average_fps = metadata.average_fps

    timestamps = [
        max(min(ts, (metadata.num_frames - 1) / average_fps), 0)
        for ts in timestamps
    ]
    frame_indices = [
        max(min(round(ts * average_fps), metadata.num_frames - 1), 0)
        for ts in timestamps
    ]

    frames_batch = decoder.get_frames_at(indices=frame_indices)

    loaded_frames = []
    loaded_ts = []

    for frame, pts in zip(frames_batch.data, frames_batch.pts_seconds, strict=False):
        loaded_frames.append(frame)
        loaded_ts.append(pts.item())

    import torch
    query_ts = torch.tensor(timestamps)
    loaded_ts = torch.tensor(loaded_ts)

    dist = torch.cdist(query_ts[:, None], loaded_ts[:, None], p=1)
    min_, argmin_ = dist.min(1)

    is_within_tol = min_ < tolerance_s
    assert is_within_tol.all(), f"Timestamp violation..."

    closest_frames = torch.stack([loaded_frames[idx] for idx in argmin_])
    closest_frames = closest_frames.type(torch.float32) / 255

    return closest_frames


video_utils.decode_video_frames_torchcodec = decode_video_frames_torchcodec_patched
