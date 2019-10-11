#%%
import pandas as pd
import tensorflow as tf

from examples.utils import build_features, get_ml_dataset
from recman import FM

tf.compat.v1.logging.set_verbosity(20)

#%%
df_train, df_test, domains = get_ml_dataset(frac=0.2)

#%%
feat_dict = build_features(pd.concat([df_train, df_test], axis=0), domains)

#%%
model = FM(
    feat_dict,
    learning_rate=0.001,
    epoch=5,
    use_interactive_session=False,
    log_dir="../logs",
)
model.fit(df_train, df_train["label"].values.reshape((-1, 1)))

#%%
print(model.predict(df_train))
