import numpy as np
import librosa

def estimate_mean_watermark_freq(y, sr, wm_band=(16000, 20000),
                                 n_fft=4096, hop_length=1024):
    """
    Estimate the mean watermark frequency by:
    - computing the STFT,
    - focusing on a high-frequency band (wm_band),
    - taking, for each time frame, the frequency bin with maximum energy in that band,
    - averaging these peak frequencies over time.
    """
    # STFT
    D = librosa.stft(y, n_fft=n_fft, hop_length=hop_length, window="hann")
    magnitude = np.abs(D)

    # Frequency axis for the STFT bins
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)

    # Select only the band where the watermark lives, e.g. 16–20 kHz
    band_mask = (freqs >= wm_band[0]) & (freqs <= wm_band[1])
    band_freqs = freqs[band_mask]

    if band_freqs.size == 0:
        raise ValueError("Watermark band is empty; check wm_band or n_fft.")

    band_mag = magnitude[band_mask, :]  # shape: [freq_in_band, time_frames]

    peak_freqs = []
    for t in range(band_mag.shape[1]):
        col = band_mag[:, t]
        if np.all(col == 0):
            continue  # no energy in this frame in the watermark band
        idx = np.argmax(col)
        peak_freqs.append(band_freqs[idx])

    if len(peak_freqs) == 0:
        return None  # watermark not detectable

    return float(np.mean(peak_freqs))


def analyze_speedup_with_watermark(path1, path2,
                                   wm_band=(16000, 20000),
                                   n_fft=4096, hop_length=1024):
    """
    Compare two audio files that contain the same watermark.
    Use the position of the watermark in frequency to:
    - estimate the mean watermark frequency in each file,
    - infer which file was time-domain slowed (lower freq),
    - infer which file was frequency-domain stretched (same freq),
    - estimate the slowdown factor x as ratio of the two means.
    """
    # Load audios with native sampling rate
    y1, sr1 = librosa.load(path1, sr=None)
    y2, sr2 = librosa.load(path2, sr=None)

    if sr1 != sr2:
        raise ValueError(f"Sampling rates differ (sr1={sr1}, sr2={sr2}). "
                         "Resample or handle separately before comparison.")

    f1 = estimate_mean_watermark_freq(y1, sr1, wm_band, n_fft, hop_length)
    f2 = estimate_mean_watermark_freq(y2, sr2, wm_band, n_fft, hop_length)

    if f1 is None or f2 is None:
        print("Could not estimate a reliable watermark frequency in one of the files.")
        print(f"file1 mean watermark freq: {f1}")
        print(f"file2 mean watermark freq: {f2}")
        return

    print(f"Estimated mean watermark frequencies:")
    print(f"  file1: {f1:.2f} Hz")
    print(f"  file2: {f2:.2f} Hz")

    # Decide which one is time-domain slowed (lower freq)
    if f1 < f2:
        time_domain_file = "file1"
        freq_domain_file = "file2"
        f_low, f_high = f1, f2
    else:
        time_domain_file = "file2"
        freq_domain_file = "file1"
        f_low, f_high = f2, f1

    x = f_high / f_low

    print()
    print(f"→ {time_domain_file} is likely the time-domain slowed file "
          f"(watermark frequency divided by x).")
    print(f"→ {freq_domain_file} is likely the frequency-domain stretched file "
          f"(watermark stays at original frequency).")
    print(f"Estimated slowdown factor: x ≈ {x:.2f}")

    return {
        "file1_mean_freq": f1,
        "file2_mean_freq": f2,
        "time_domain_file": time_domain_file,
        "freq_domain_file": freq_domain_file,
        "x_estimated": x,
    }

# Exemple d'utilisation :
# result = analyze_speedup_with_watermark("task3_file1.wav", "task3_file2.wav",
#                                         wm_band=(16000, 20000))
