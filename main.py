from pathlib import Path
import torch
import torchvision
import typer
from PIL import Image
from tqdm.auto import tqdm

def phase_correlate_vertical(img1, img2, subsample_horiz=25):
    """Find vertical offset between two images using phase correlation"""
    # FFT both images
    img1 = img1[..., ::subsample_horiz]
    img2 = img2[..., ::subsample_horiz]
    F1 = torch.fft.fft2(img1)
    F2 = torch.fft.fft2(img2)

    # Cross-power spectrum
    cross_power = F1 * torch.conj(F2)
    cross_power /= torch.abs(cross_power) + 1e-10

    # Inverse FFT gives correlation surface
    correlation = torch.fft.ifft2(cross_power).real

    # Find peak location
    peak_idx = torch.unravel_index(torch.argmax(correlation), correlation.shape)
    vertical_offset = peak_idx[0].item()

    # Handle wrap-around (negative offsets appear at end)
    h = img1.shape[0]
    if vertical_offset > h // 2:
        vertical_offset -= h

    return vertical_offset

def main(video_path = Path("./ScreenRecording_06-11-2025 13-53-32_1.MP4")):

    video_reader = torchvision.io.VideoReader(video_path)
    px_offsets = []
    CROP_TOP = 512
    N_FRAMES = 550
    prev_frame = None

    for frame in tqdm(video_reader, total=N_FRAMES, ncols=60):
        # Process each frame
        frame = frame['data'] / 255.0  # Normalize the frame
        frame = frame[:, CROP_TOP:-CROP_TOP]
        frame = frame.mean(dim=0, keepdim=True)  # Convert to grayscale

        if prev_frame is None:
            prev_frame = frame
            px_offsets.append(0)
        else:
            # Use phase correlation to find vertical offset
            offset = phase_correlate_vertical(frame.squeeze(), prev_frame.squeeze())
            px_offsets.append(-offset)  # Flip sign for downward scrolling
            prev_frame = frame



    # Convert relative offsets to absolute positions
    px_offsets = torch.tensor(px_offsets, dtype=torch.float32)
    absolute_positions = px_offsets.cumsum(dim=0)

    # Create panorama canvas
    min_pos = absolute_positions.min().int()
    max_pos = absolute_positions.max().int()
    frame_height = frame.shape[1]
    total_height = max_pos - min_pos + frame_height
    total_width = frame.shape[2]

    panorama = torch.zeros((3, total_height, total_width), dtype=torch.float32)
    count = torch.zeros((total_height, total_width), dtype=torch.float32)

    # Stitch frames into panorama
    print(frame.shape)
    video_reader = torchvision.io.VideoReader(video_path)
    for i, (frame, pos) in enumerate(zip(video_reader, absolute_positions)):
        frame = frame['data'] / 255.0  # Normalize the frame
        frame = frame[:, CROP_TOP:-CROP_TOP]

        start_row = int(pos - min_pos)
        end_row = start_row + frame_height
        panorama[:, start_row:end_row] += frame
        count[start_row:end_row] += 1

    # Average overlapping regions
    panorama = panorama / (count + 1e-10)

    # Normalize for display
    panorama -= panorama.min()
    panorama /= panorama.max()

    Image.fromarray((panorama.permute(1, 2, 0).numpy() * 255).astype('uint8')).save("output.png")

    #import IPython; IPython.embed()




if __name__ == "__main__":
    typer.run(main)
