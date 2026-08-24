import argparse
import glob
import json
import os
import time

import librosa
import numpy as np
import soundfile as sf
import torch

from dataset import mel_spectrogram
from utils import AttrDict


def load_checkpoint(filepath, device):
    assert os.path.isfile(filepath)
    print(f"Loading '{filepath}'")
    checkpoint_dict = torch.load(filepath, map_location=device)
    print("Complete.")
    return checkpoint_dict


def get_mel(x, h):
    return mel_spectrogram(
        x,
        h.n_fft,
        h.num_mels,
        h.sampling_rate,
        h.hop_size,
        h.win_size,
        h.fmin,
        h.fmax,
    )


def scan_checkpoint(cp_dir, prefix):
    pattern = os.path.join(cp_dir, prefix + '*')
    cp_list = glob.glob(pattern)
    if len(cp_list) == 0:
        return ''
    return max(cp_list)


def inference(h, device):
    model_module = __import__(h.model_module, fromlist=["Generator"])
    Generator = model_module.Generator
    generator = Generator(h).to(device)

    state_dict_g = load_checkpoint(h.checkpoint_file_load, device)
    generator.load_state_dict(state_dict_g['generator'])

    input_dir = h.test_input_mels_dir if h.test_mel_load else h.test_input_wavs_dir
    filelist = sorted(os.listdir(input_dir))

    os.makedirs(h.test_output_dir, exist_ok=True)

    generator.eval()
    total_samples = 0
    with torch.no_grad():
        starttime = time.time()
        for filename in filelist:
            input_path = os.path.join(input_dir, filename)
            output_length = None
            if h.test_mel_load:
                mel = np.load(input_path)
                x = torch.as_tensor(mel, dtype=torch.float32, device=device)
                if x.ndim == 2:
                    x = x.unsqueeze(0)
                if x.shape[-1] == h.num_mels:
                    x = x.transpose(1, 2)
                if x.ndim != 3 or x.shape[1] != h.num_mels:
                    raise ValueError(
                        f"Expected mel input shaped [B, {h.num_mels}, T] or [B, T, {h.num_mels}], "
                        f"got {tuple(x.shape)} from {input_path}"
                    )
            else:
                raw_wav, _ = librosa.load(input_path, sr=h.sampling_rate, mono=True)
                raw_wav = torch.as_tensor(raw_wav, dtype=torch.float32, device=device)
                x = get_mel(raw_wav.unsqueeze(0), h)
                output_length = raw_wav.numel()

            if h.test_mel_load:
                output_length = getattr(h, "inference_length", None)
            _logamp_g, _pha_g, _, _, y_g = generator(x, length=output_length)
            audio = y_g.squeeze()
            audio = audio.cpu().numpy()
            output_name = os.path.splitext(filename)[0] + ".wav"
            sf.write(
                os.path.join(h.test_output_dir, output_name),
                audio,
                h.sampling_rate,
                "PCM_16",
            )
            total_samples += len(audio)

        end=time.time()
        print(end-starttime)
        print(total_samples / h.sampling_rate)
        print(total_samples / h.sampling_rate / (end-starttime))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--model", choices=("apnet2", "freev"), default="apnet2")
    parser.add_argument(
        "--length",
        dest="inference_length",
        type=int,
        default=None,
        help="Output sample length for precomputed mel input.",
    )
    args = parser.parse_args()

    print("Initializing Inference Process..")

    with open(args.config) as f:
        data = f.read()

    json_config = json.loads(data)
    h = AttrDict(json_config)
    h.model_module = "models" if args.model == "apnet2" else "models_freev"
    h.inference_length = args.inference_length

    torch.manual_seed(h.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(h.seed)
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    inference(h, device)


if __name__ == '__main__':
    main()

