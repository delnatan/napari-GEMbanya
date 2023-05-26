import numpy as np
import pandas as pd
from laptrack import LapTrack
from nd2reader import ND2Reader
from scipy.optimize import minimize
from skimage.feature import peak_local_max
from scipy.ndimage import gaussian_laplace
from tqdm import tqdm


def read_timelapse(img):
    """convenient function to read ND2 images"""

    def _fetch_camera_header(reader):
        hdrstr = reader.parser._raw_metadata.image_text_info[
            b"SLxImageTextInfo"
        ][b"TextInfoItem_6"].decode()
        return hdrstr

    with ND2Reader(img) as i:
        i.bundle_axes = "tyx"
        pixel_size = i.metadata["pixel_microns"]
        headerstr = _fetch_camera_header(i)
        imgdata = i[0]

        return imgdata, headerstr, pixel_size


def peaks2d_to_boxes(peaklist, boxsize=5):
    """generate square box coordinates for adding to napari

    Usage:
    peaklist (np.array): array with 3 columns (z-index,
    y, x) for 2d peak locations per z-slice
    boxsize (int): size of the box surround peak


    ..code-block :: python

        viewer.add_shapes(
            boxcoords,
            shape_type="rectangle",
            edge_width=0.5,
            edge_color="yellow",
            face_color="transparent",
            name="peaks"
        )

    """
    boxcoords = []
    w = boxsize // 2

    for p in peaklist:
        zid, yc, xc = p
        box_ = np.array(
            [
                [zid, yc - w - 0.5, xc - w - 0.5],
                [zid, yc - w - 0.5, xc + w + 0.5],
                [zid, yc + w + 0.5, xc + w + 0.5],
                [zid, yc + w + 0.5, xc - w - 0.5],
            ]
        )
        boxcoords.append(box_)

    return boxcoords


def _correct_angle(theta):
    """adjust np.angle output to map from 0 to 2pi"""
    if theta > 0:
        return theta - 2 * np.pi
    else:
        return theta


def get_offset_with_phasor(roi):
    """use phasor method to refine maxima coordinates

    Input:
    roi (2-d numpy.array): roi of fluorescent spot. Size must be odd-numbered and square

    Returns:
    shift in y-axis, shift in x-axis

    """
    # size of window
    N = roi.shape[0]
    twopi = 2.0 * np.pi
    roi_ft = np.fft.rfft2(roi)
    # take the first Fourier coefficient along each axes
    ycoef1 = roi_ft[1, 0]
    xcoef1 = roi_ft[0, 1]
    # compute angles from fourier coefficient (and correct it)
    y_angle = _correct_angle(np.angle(ycoef1))
    x_angle = _correct_angle(np.angle(xcoef1))
    # normalize angle by 2*pi/N (the max frequency)
    y_shift = np.abs(y_angle) / (twopi / N)
    x_shift = np.abs(x_angle) / (twopi / N)
    return y_shift - N // 2, x_shift - N // 2


def rayleigh_pdf(x, D, slope, dt, r_max):
    a = 4 * D * dt
    prob = ((2 * x) / a) * np.exp(-(x * x) / a) + slope * x
    Z = 1 - np.exp(-(r_max * r_max) / a) + (r_max * r_max) * slope / 2.0
    return prob / Z


def mle_fit(data, init_D=1.0, init_b=1e-2, r_max=0.8, dt=0.020):
    def _obj(x, *args):
        diffusion_coef = x[0]
        slope = x[1]
        data, r_max, dt = args

        a = 4 * dt * diffusion_coef
        prob = ((2 * data) / a) * np.exp(-(data * data) / a) + slope * data
        Z = 1 - np.exp(-(r_max * r_max) / a) + (r_max * r_max) * slope / 2.0
        pdf = prob / Z
        logprob = np.log(pdf)

        # minimize negative log-likehood
        return -np.sum(logprob)

    p0 = (init_D, init_b)
    res = minimize(
        _obj,
        p0,
        args=(data, r_max, dt),
        method="Nelder-Mead",
        bounds=((1e-6, None), (0.0, None)),
    )

    return {"D": res.x[0], "slope": res.x[1]}


def isolate_spots(data, mask=None, threshold=10, sigma=1.5):
    """uses 2d laplace operator to identify fluorescent peaks"""
    Nt, Ny, Nx = data.shape

    spots = []

    for t in tqdm(range(Nt)):
        _frame = np.float64(data[t, :, :])
        _response = -gaussian_laplace(_frame, sigma)
        _peaklocs = peak_local_max(
            _response, labels=mask, threshold_abs=threshold, min_distance=3
        )
        _df = pd.DataFrame(_peaklocs, columns=["y", "x"])
        _df["frame"] = t
        spots.append(_df)

    all_spots = pd.concat(spots)

    return all_spots.reset_index(drop=True)


def refine_spots(img, spotlocs, boxsize=7):
    """refine spot coordinates using phasor method"""
    yid = spotlocs["y"].to_numpy().astype(int)
    xid = spotlocs["x"].to_numpy().astype(int)
    tid = spotlocs["frame"].to_numpy().astype(int)
    d = boxsize // 2
    ref_y = np.zeros(len(yid))
    ref_x = np.zeros(len(xid))
    for n in tqdm(range(len(yid))):
        _y = yid[n]
        _x = xid[n]
        _t = tid[n]
        roi = img[_t, _y - d : _y + d + 1, _x - d : _x + d + 1]
        dy, dx = get_offset_with_phasor(roi)
        ref_y[n] = _y + dy
        ref_x[n] = _x + dx
        spotlocs["x"] = ref_x
        spotlocs["y"] = ref_y


def link_spots_to_trajectory(spots, max_displacement=10.0):
    """run LapTrack algorithm on identified spot coordinates

    spots (DataFrame): with columns "x", "y", "frame" at least
    max_displacement (optional, float): maximum displacement in pixels

    """
    tracker = LapTrack(
        track_cost_cutoff=max_displacement**2, gap_closing_cost_cutoff=False
    )
    track_df, split_df, merge_df = tracker.predict_dataframe(
        spots,
        coordinate_cols=["x", "y"],
        frame_col="frame",
        only_coordinate_cols=False,
        validate_frame=False,
    )
    track_df = track_df.reset_index()
    return track_df


def get_paired_displacements(linked_spots):
    """compute displacements between pairs of coordinates from frames
    (t, t+1).

    linked_spots (DataFrame): spot coordinates after trajectory linking. Must
    have columns "x" and "y".

    Returns:
    np.array containing all displacements from a single timeframe

    """
    # cast 'frame' column as integer (if not already!)
    linked_spots["frame"] = linked_spots["frame"].astype(int)
    Nframes = linked_spots["frame"].max()
    pwdistances = []

    for n in range(Nframes - 1):
        # get a pair of frames from the current one
        start_frame = linked_spots[linked_spots["frame"] == n]
        end_frame = linked_spots[linked_spots["frame"] == n + 1]
        # only get track_id that is paired between these two frames
        valid_track_id = list(
            set(start_frame["track_id"]) & set(end_frame["track_id"])
        )
        # filter the start and endpoint frames according to the valid_track_id
        valid_start = start_frame[start_frame["track_id"].isin(valid_track_id)]
        valid_end = end_frame[end_frame["track_id"].isin(valid_track_id)]
        # since track_id is unique we can use to as an index
        valid_start.set_index(valid_start["track_id"], inplace=True)
        valid_end.set_index(valid_end["track_id"], inplace=True)
        # we want to only calculate distance of the same track_id
        # now made easier due to index-matched Series operation
        _pwdist = np.sqrt(
            np.square(valid_start[["y", "x"]] - valid_end[["y", "x"]]).sum(
                axis=1
            )
        )
        pwdistances.extend(_pwdist.tolist())

    return np.array(pwdistances)


def compute_msd(sdf):
    """compute MSD for single trajectory

    Args:
    sdf (DataFrame): with columns 'x_um','y_um','frame' and single 'track_id'.

    Returns:
    DataFrame with 'lag', 'MSD', 'stdMSD', 'n'

    here 'n' contains the number of points used in averaging distances

    Usage:
    tracks.groupby("track_id").apply(compute_msd).reset_index(level=0)

    """
    nframes = sdf.shape[0]

    if nframes > 1:
        lags = np.arange(1, nframes)
        nlags = len(lags)
        msdarr = np.zeros(nlags)
        stdarr = np.zeros(nlags)
        npts = np.zeros(nlags, dtype=int)
        frames = sdf["frame"].to_numpy()
        frames_set = set(frames)
        xc = sdf["x"].to_numpy()
        yc = sdf["y"].to_numpy()

        for i, lag in enumerate(lags):
            # ensure only correct lag time is used
            # find frames that have the current lag length
            frames_lag_end = frames + lag
            valid_end_frames = np.array(list(frames_set & set(frames_lag_end)))
            valid_start_frames = valid_end_frames - lag
            s1 = np.where(np.isin(frames, valid_start_frames))[0]
            s2 = np.where(np.isin(frames, valid_end_frames))[0]
            # only take distances that correspond to the correct lag
            sqdist = np.square(xc[s2] - xc[s1]) + np.square(yc[s2] - yc[s1])
            msdarr[i] = np.mean(sqdist)
            stdarr[i] = np.std(sqdist)
            npts[i] = len(sqdist)

        return pd.DataFrame(
            {"lag": lags, "MSD": msdarr, "stdMSD": stdarr, "n": npts}
        )
    else:
        return None


def fitline(x, y, s):
    n = len(x)
    s2 = s * s
    wcovxy = np.sum((x * y) / s2)
    wvarx = np.sum((x * x) / s2)
    slope = wcovxy / wvarx
    mse = np.sum(np.square(y - slope * x)) / (n - 1)
    return slope, mse


def fitline2(x, y, s):
    """https://chem.libretexts.org/Bookshelves/Analytical_Chemistry/Chemometrics_Using_R_(Harvey)/08%3A_Modeling_Data/8.02%3A_Weighted_Linear_Regression_with_Errors_in_y"""
    n = len(x)
    w = (n / s**2) / np.sum(1 / s**2)
    wy = np.sum(w * y)
    wx = np.sum(w * x)
    wxy = np.sum(w * x * y)
    wxx = np.sum(w * x * x)

    slope = (n * wxy - wx * wy) / (n * wxx - wxx**2)
    intercept = (wy - slope * wx) / n

    return slope, intercept
