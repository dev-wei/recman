import numpy as np
import pandas as pd


def gini(y_true, y_score):
    assert len(y_true) == len(y_score)

    all_together = np.asarray(
        np.c_[y_true, y_score, np.arange(len(y_score))], dtype=np.float
    )
    all_together = all_together[np.lexsort((all_together[:, 2], -1 * all_together[:1]))]
    total_losses = all_together[:, 0].sum()
    gini_sum = all_together[:, 0].cumsum().sum() / total_losses
    gini_sum -= (len(y_true) + 1) / 2
    return gini_sum / len(y_true)


def gini_norm(y_true, y_score):
    return gini(y_true, y_score) / gini(y_true, y_true)


def prepare_test_data(feat_dict, df_X):
    user_feats, class_feats, aux_feats = [], [], []

    for feat in feat_dict.values():
        if feat.source == "USER":
            user_feats.append(feat)
        elif feat.source == "CLASS":
            class_feats.append(feat)
        elif feat.source == "AUX":
            aux_feats.append(feat)
        else:
            pass

    user_attrs = dict()
    user_dict = df_X.groupby("USER_ID").groups
    for feat in user_feats:
        for user_id, indices in user_dict.items():
            if user_id not in user_attrs:
                user_attrs[user_id] = dict()
            val = df_X.iloc[indices,][feat.name].unique()
            user_attrs[user_id][feat.name] = val[0] if len(val) == 1 else set(val)

    class_attrs = dict()
    class_dict = df_X.groupby("CLASS_ID").groups
    for feat in class_feats:
        for class_id, indices in class_dict.items():
            if class_id not in class_attrs:
                class_attrs[class_id] = dict()
            val = df_X.iloc[indices,][feat.name].unique()
            class_attrs[class_id][feat.name] = val[0] if len(val) == 1 else set(val)

    aux_attrs = dict()
    aux_dict = df_X.groupby(["USER_ID", "CLASS_ID"]).groups
    for feat in aux_feats:
        for (user_id, class_id), indices in aux_dict.items():
            if (user_id, class_id) not in aux_attrs:
                aux_attrs[(user_id, class_id)] = dict()
            val = df_X.iloc[indices,][feat.name].unique()
            aux_attrs[(user_id, class_id)][feat.name] = (
                val[0] if len(val) == 1 else set(val)
            )

    return user_attrs, class_attrs, aux_attrs


def top_k_precision(feat_dict: dict, predict, df_X_train, df_X_test_true, K=5):
    user_attrs_train, _, _ = prepare_test_data(feat_dict, df_X_train)
    user_attrs, class_attrs, aux_attrs = prepare_test_data(
        feat_dict, pd.concat([df_X_train, df_X_test_true]).reset_index()
    )

    n_users = feat_dict["USER_ID"].feat_size
    n_classes = feat_dict["CLASS_ID"].feat_size

    users_in_test = set(df_X_test_true["USER_ID"].unique())
    users_in_train = set(df_X_train["USER_ID"].unique())

    hot_users = {user_id for user_id in users_in_test if user_id in users_in_train}
    cold_users = users_in_test - hot_users

    # hot user assessment
    scores = dict()
    for zone, users in (("hot_users", hot_users), ("cold_users", cold_users)):
        relevant, total = 0, 0
        for user_id in users:
            df_X = pd.DataFrame(columns=[feat for feat in feat_dict])

            taken_class_ids = user_attrs_train[user_id]["CLASS_ID"]
            un_taken_class_ids = [
                class_id
                for class_id in range(1, n_classes)
                if class_id not in taken_class_ids
            ]
            df_X["USER_ID"] = [user_id] * len(un_taken_class_ids)
            df_X["CLASS_ID"] = un_taken_class_ids
            for feat in feat_dict.values():
                if feat.name not in ["USER_ID", "CLASS_ID"]:
                    if feat.source == "USER":
                        df_X[feat.name] = [user_attrs[user_id][feat.name]] * len(
                            un_taken_class_ids
                        )
                    elif feat.source == "CLASS":
                        df_X[feat.name] = [
                            class_attrs[class_id][feat.name]
                            for class_id in un_taken_class_ids
                        ]
                    elif feat.source == "AUX":
                        df_X[feat.name] = [
                            aux_attrs[f"{user_id},{class_id}"][feat.name]
                            for class_id in un_taken_class_ids
                        ]

            pred = sorted(zip(df_X["CLASS_ID"], predict(df_X)), key=lambda x: -x[1])[:K]
            good_choices = set(
                df_X_test_true[df_X_test_true.USER_ID == user_id].CLASS_ID
            )

            total += min(K, len(good_choices))
            for class_id, _ in pred:
                if class_id in good_choices:
                    relevant += 1

        scores[zone] = relevant / total if total > 0 else 0

    return scores
