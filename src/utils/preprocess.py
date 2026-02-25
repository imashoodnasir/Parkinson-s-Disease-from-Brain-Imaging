from typing import Tuple
import os
import numpy as np
import nibabel as nib
import SimpleITK as sitk

def n4_bias_correction(img: sitk.Image) -> sitk.Image:
    mask = sitk.OtsuThreshold(img, 0, 1, 200)
    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    return corrector.Execute(img, mask)

def resample_to_spacing(img: sitk.Image, out_spacing: Tuple[float,float,float]) -> sitk.Image:
    orig_spacing = img.GetSpacing()
    orig_size = img.GetSize()
    out_size = [int(round(osz*ospc/nspc)) for osz, ospc, nspc in zip(orig_size, orig_spacing, out_spacing)]
    resampler = sitk.ResampleImageFilter()
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetOutputSpacing(out_spacing)
    resampler.SetSize(out_size)
    resampler.SetOutputDirection(img.GetDirection())
    resampler.SetOutputOrigin(img.GetOrigin())
    return resampler.Execute(img)

def center_crop_or_pad(vol: np.ndarray, target_size: Tuple[int,int,int]) -> np.ndarray:
    tz, ty, tx = target_size
    z, y, x = vol.shape
    out = np.zeros((tz, ty, tx), dtype=np.float32)

    def _compute(in_len, out_len):
        if in_len >= out_len:
            s_in = (in_len - out_len)//2
            e_in = s_in + out_len
            s_out, e_out = 0, out_len
        else:
            s_in, e_in = 0, in_len
            s_out = (out_len - in_len)//2
            e_out = s_out + in_len
        return s_in, e_in, s_out, e_out

    zin0, zin1, zout0, zout1 = _compute(z, tz)
    yin0, yin1, yout0, yout1 = _compute(y, ty)
    xin0, xin1, xout0, xout1 = _compute(x, tx)
    out[zout0:zout1, yout0:yout1, xout0:xout1] = vol[zin0:zin1, yin0:yin1, xin0:xin1]
    return out

def zscore_normalize(vol: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    return (vol - float(vol.mean())) / (float(vol.std()) + eps)

def preprocess_nifti(
    in_path: str,
    out_path: str,
    target_spacing: Tuple[float,float,float] = (1.0,1.0,1.0),
    target_size: Tuple[int,int,int] = (128,128,128),
    use_n4: bool = False,
) -> None:
    nimg = nib.load(in_path)
    data = np.asarray(nimg.get_fdata(), dtype=np.float32)

    # Convert to SimpleITK in z,y,x order for consistent resampling
    sitk_img = sitk.GetImageFromArray(data.astype(np.float32))
    zooms = nimg.header.get_zooms()[:3]
    sitk_img.SetSpacing((float(zooms[0]), float(zooms[1]), float(zooms[2])))

    if use_n4:
        sitk_img = n4_bias_correction(sitk_img)

    sitk_img = resample_to_spacing(sitk_img, target_spacing)
    arr = sitk.GetArrayFromImage(sitk_img).astype(np.float32)  # z,y,x

    arr = zscore_normalize(arr)
    arr = center_crop_or_pad(arr, target_size)

    out_img = nib.Nifti1Image(arr, affine=nimg.affine)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    nib.save(out_img, out_path)
