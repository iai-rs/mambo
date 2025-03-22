total_epochs = 1000000
save_every = 2000
learning_rate = 5e-5
batch_size = 8
image_size = 256
optimizer_type = 'adam'
total_timesteps = 1000
channels = 2
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
exp_name = 'vindr_unified_4x'

server='fmle'
if server == 'fmle':
    WH_PATH = '/lustre/mambo/models/artifacts/vindr_healthy_256:v82/model_124499.pt'
    LC_PATH = '/lustre/mambo/models/artifacts/vindr_lcl_ctx_3072:v37/model_56999.pt'
    PH_PATH = '/lustre/mambo/models/artifacts/vindr_3c_256_v2:v84/model_169999.pt'
    SAVE_DIR = '/lustre/mambo/results/'
elif server == 'mambo':
    WH_PATH = '/mambo/artifacts/vindr_healthy_256:v82/model_124499.pt'
    LC_PATH = '/mambo/artifacts/vindr_lcl_ctx_3072:v37/model_56999.pt'
    PH_PATH = '/mambo/artifacts/vindr_3c_256_v2:v84/model_169999.pt'
    SAVE_DIR = '/mambo/data/'
