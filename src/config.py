total_epochs = 132000
save_every = 2000
learning_rate = 5e-5
batch_size = 8
image_size = 256
optimizer_type = 'adam'
total_timesteps = 1000
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


server='fmle'
if server == 'fmle':
    artifacts_dir = '/home/milica.skipina.ivi/projects/test_git/mambo/models/artifacts'
    SAVE_DIR = '/lustre/seed/vindr_birads_cond/'
elif server == 'mambo':
    artifacts_dir = '/mambo/artifacts'
    SAVE_DIR = '/mambo/data/'

WH_PATH = f'{artifacts_dir}/vindr_healthy_256:v82/model_124499.pt'
LC_PATH = f'{artifacts_dir}/vindr_lcl_ctx_3072:v37/model_56999.pt'
PH_PATH = f'{artifacts_dir}/vindr_3c_256_v2:v84/model_169999.pt'

RSNA_WH_PATH = f'{artifacts_dir}/rsna_machine49_256:v178/model_268499.pt'
RSNA_LC_PATH = f'{artifacts_dir}/rsna_3c_local_ctx_m49:v14/model_22499.pt'
RSNA_PH_PATH = f'{artifacts_dir}/rsna_3c_256:v70/model_131999.pt'
