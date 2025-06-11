from pathlib import Path
import torch
import torchvision
import typer
from PIL import Image
from tqdm.auto import tqdm

def main(video_path = Path("./ScreenRecording_06-11-2025 13-53-32_1.MP4")):

    video_reader = torchvision.io.VideoReader(video_path)
    signatures_by_row = []
    # px_offsets = []
    CROP_TOP = 128
    prev_frame = None
    for frame in tqdm(video_reader, total=500, ncols=60):
        # Process each frame (this is just a placeholder)
        frame = frame['data'] / 255.0  # Normalize the frame
        frame = frame[:, CROP_TOP:-CROP_TOP]
        frame = frame.mean(dim=0, keepdim=True)
        frame = frame - frame.mean(dim=2, keepdim=True)
        frame /= (0.001 + frame.std(dim=2, keepdim=True))
        SCALE = 20
        frame = frame[:, :, ::SCALE] # downsample width

        if prev_frame is None:
            prev_frame = frame

        matches = torch.nn.functional.conv2d(
            frame[None],
            prev_frame[None],
            padding=(128, 0),
        )
        # import IPython; IPython.embed()
        signatures_by_row.append(matches.squeeze())
        prev_frame = frame



    #     # Per-row signatures: difference of consecutive pixels
    #     print(frame.shape)
    #     raise ValueError()
    #     signatures = (frame[:-1, 1:] - frame[:-1, :-1] - frame[1:, :-1] + frame[1:,1:]).mean(dim=1)

    #     # Calculate the closest pixel offset
    #     if signatures_by_row:
    #         a = signatures [None,None]
    #         b = signatures_by_row[-1][None,None]
    #         print(b.shape)
    #         b = b.flip(dims=(-1,))
    #         PADDING = 128
    #         offset = torch.nn.functional.conv1d(a,b, padding=PADDING)
    #         print(offset[0,0,120:136])
    #         offset = PADDING - offset.argmax()
    #         print(offset)
    #         px_offsets.append(offset)
    #     signatures_by_row.append(signatures)

    # px_offsets = torch.stack(px_offsets)
    # # relative to absolute
    # px_offsets = px_offsets.cumsum(dim=0)
    # total_n_rows = px_offsets.max()
    # img = torch.zeros((total_n_rows, len(signatures_by_row)), dtype=torch.float32)
    # for i, (offset, sig) in enumerate(zip(px_offsets, signatures_by_row)):
    #     # Place the signature in the correct row
    #     W = min(len(sig), img.shape[0] - offset)
    #     img[offset:(offset+W), i] = sig[:W]


    img = torch.stack(signatures_by_row)

    img -= img.min()
    img /= img.max()
    Image.fromarray(img.numpy() * 255).convert("L").save("output.png")

    #import IPython; IPython.embed()




if __name__ == "__main__":
    typer.run(main)
