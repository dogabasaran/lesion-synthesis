import numpy as np
import nibabel as nib
import os
import scipy.ndimage.measurements as measure 
import torch 

def remove_small_cc(binary, thres=10):
    """ Remove small connected component in the foreground. """
    cc, n_cc = measure.label(binary)
    binary2 = np.copy(binary)
    for n in range(1, n_cc + 1):
        area = np.sum(cc == n)
        if area < thres:
            binary2[cc == n] = 0
    return np.float32(np.array(binary2.astype(int)))

def register(to_this, register_this, output_path):
    os.system(f'mirtk register {register_this} {to_this} '   
                f'-parin masking/ffd.cfg -output {output_path} -v 0')
    
def get_brain_mask(img):
    os.system(f'bet {img} {img} -m -n')   # generates brain mask, saved as "<img>_mask.nii.gz"


def retrieve_mask(real_A, fake_B): 
    """
    input: real_A and fake_B
    output: lesion mask from fake_B

    use the registered healthy brain to grab the brain and White matter mask.

    """
    if not os.path.exists('masking/temp_data'):
        os.makedirs('masking/temp_data')

    real_A = np.rot90(real_A.cpu().detach().numpy().squeeze(), k=3)
    fake_B = np.rot90(fake_B.cpu().detach().numpy().squeeze(), k=3)
    realA_path = "masking/temp_data/realA.nii.gz"
    fakeB_path = "masking/temp_data/fakeB.nii.gz"
    real_A = nib.Nifti1Image(real_A[None, :], affine=np.eye(4)) 
    fake_B = nib.Nifti1Image(fake_B[None, :], affine=np.eye(4))
    nib.save(real_A, realA_path)
    nib.save(fake_B, fakeB_path)


 
    # register the healthy (realA) to the unhealthy (fakeB)
    register(realA_path, fakeB_path, realA_path.replace('.nii.gz', '_regged.nii.gz'))

    # Obtain brain mask -> saved as real_A_regged_mask.nii.gz
    get_brain_mask(realA_path.replace('.nii.gz', '_regged.nii.gz'))  

    # Multiply real_A_regged x real_A_regged_mask --> real_A_brain
    os.system(f'fslmaths {realA_path.replace(".nii.gz", "_regged.nii.gz")} -mul {realA_path.replace(".nii.gz", "_regged_mask.nii.gz")} {realA_path.replace(".nii.gz", "_brain.nii.gz")}')

    # Multilpy fake_B x real_A_regged_mask --> fake_B_brain
    os.system(f'fslmaths {fakeB_path} -mul {realA_path.replace(".nii.gz", "_regged_mask.nii.gz")} {fakeB_path.replace(".nii.gz", "_brain.nii.gz")}')

    os.system(f'fast -t 2 -n 2 -H 0.1 -I 4 -l 20.0 -o {realA_path.replace(".nii.gz", "_brain")} {realA_path.replace(".nii.gz", "_brain")}')


    realA = nib.load(realA_path.replace(".nii.gz", "_brain.nii.gz")).get_fdata().squeeze()
    fakeB = nib.load(fakeB_path.replace(".nii.gz", "_brain.nii.gz")).get_fdata().squeeze()
    realawmm = nib.Nifti1Image(realA[None, :], affine=np.eye(4))
    nib.save(realawmm, realA_path.replace('.nii.gz', '_wmm.nii.gz'))
    fakebwmm = nib.Nifti1Image(fakeB[None, :], affine=np.eye(4))
    nib.save(fakebwmm, fakeB_path.replace('.nii.gz', '_wmm.nii.gz'))


    difference = ((fakeB) - (realA))

    diff = (difference > .075).astype(int).astype(np.float64())
    fake_C = remove_small_cc(diff)
    diff = nib.Nifti1Image(difference[None, :], affine=np.eye(4))
    nib.save(diff, fakeB_path.replace('.nii.gz', '_soft_seg.nii.gz'))
    nfake_C = nib.Nifti1Image(fake_C[None, :], affine=np.eye(4))
    nib.save(nfake_C, "masking/temp_data/fake_C.nii.gz")
    fake_C = np.rot90(fake_C, k=1)
    fake_C = fake_C[None, None, :]

    os.system('rm -r masking/temp_data')

    return torch.from_numpy(fake_C.copy()).float()


if __name__ == "__main__":

    realA_path = "1000213_ra.nii.gz"     #path to network outputs
    fakeB_path = "1000213_fb.nii.gz"           

    rA = nib.load(realA_path).get_fdata().squeeze()
    fB = nib.load(fakeB_path).get_fdata().squeeze()
    

    fake_C = retrieve_mask(rA, fB)
    print(fake_C.shape)
    print(fake_C.max())
    nfake_C = nib.Nifti1Image(fake_C[None, :], affine=np.eye(4))
    nib.save(nfake_C, "fake_C.nii.gz")


    