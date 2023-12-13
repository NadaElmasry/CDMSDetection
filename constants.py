import seaborn as sns
SEED = 42
cmap = 'coolwarm'
cmap = sns.color_palette(cmap, as_cmap=True)
k_folds = 5
scoring_metric = 'f1'
num_cols = ['Age', 'Schooling']

cat_cols = ['Gender', 'Breastfeeding', 'Varicella',
            'Initial_Symptom', 'Mono_or_Polysymptomatic', 'Oligoclonal_Bands',
            'LLSSEP', 'ULSSEP', 'VEP', 'BAEP', 'Periventricular_MRI',
            'Cortical_MRI', 'Infratentorial_MRI', 'Spinal_Cord_MRI']

gender = {1: 'Male', 2: 'Female'}
breastfeeding = {1: 'yes', 2: 'no', 3: 'unknown'}
varicella = {1: 'positive', 2: 'negative', 3: 'unknown'}
mono_polysymptomatic = {1: 'monosymptomatic',
                        2: 'polysymptomatic', 3: 'unknown'}
oligoclonal_bands = {0: 'negative', 1: 'positive', 2: 'unkown'}
llssep = {0: 'negative', 1: 'positive'}
ulssep = {0: 'negative', 1: 'positive'}
vep = {0: 'negative', 1: 'positive'}
baep = {0: 'negative', 1: 'positive'}
per_mri = {0: 'negative', 1: 'positive'}
cor_mri = {0: 'negative', 1: 'positive'}
infra_mri = {0: 'negative', 1: 'positive'}
spinal_mri = {0: 'negative', 1: 'positive'}
group = {1: 'CDMS', 0: 'Non-CDMS'}
