total_epochs = 132000
save_every = 2000
learning_rate = 5e-5
batch_size = 8
image_size = 256
optimizer_type = 'adam'
timesteps = 1000
channels = 3
input_dim = 128
dim_mults = (1, 2, 2, 4, 4)

dataset='vindr'
if dataset == "rsna":
    ORIG_IMG_SIZE = 3840
elif dataset == "vindr":
    ORIG_IMG_SIZE = 3072
else:
    ORIG_IMG_SIZE = 0
PATCH_REAL_SIZE = 256
PATCH_SCALE_FACTOR = 3
LOCAL_CONTEXT_SIZE = PATCH_REAL_SIZE * PATCH_SCALE_FACTOR
MID_IMAGE_SIZE = ORIG_IMG_SIZE // PATCH_SCALE_FACTOR
LOCAL_CONTEXT_SCALE_FACTOR = MID_IMAGE_SIZE // PATCH_REAL_SIZE
FINAL_IMAGE_SIZE = PATCH_REAL_SIZE * PATCH_SCALE_FACTOR * LOCAL_CONTEXT_SCALE_FACTOR
IS_COND = False
OVERLAP=0.125
