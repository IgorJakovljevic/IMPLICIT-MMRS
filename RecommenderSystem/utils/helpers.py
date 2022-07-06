
from typing import List
import implicit
import pandas as pd
import numpy as np
import scipy.sparse as sparse

from implicit import evaluation
from .eval import EvalResults
from .confidence_measures import MeasureOfConfidence, SimpleMeasureOfConfidence, BinaryMeasureOfConfidence, LogMeasureOfConfidence
from pathlib import Path


def save_pckl(file_path, df):
    output_file = Path(file_path)
    output_file.parent.mkdir(exist_ok=True, parents=True)
    df.to_pickle(file_path)


def calculateSparsity(df):
    # Create lists of all users, artists and plays
    users = list(np.sort(df.u_id.unique()))
    channels = list(np.sort(df.c_id.unique()))
    interactions = list(df.score)

    sparsity = (1 - (len(interactions)/(len(users)*len(channels))))*100

    user_sparsity_df = df.copy()
    user_sparsity_df = user_sparsity_df.groupby(["u_id"]).count()
    user_sparsity_max = user_sparsity_df.c_id.max()
    user_sparsity_df['USS'] = (
        1 - (user_sparsity_df["c_id"]/user_sparsity_max)) * 100

    item_sparsity_df = df.copy()
    item_sparsity_df = item_sparsity_df.groupby(["c_id"]).count()
    item_rating_max = item_sparsity_df.u_id.max()
    item_sparsity_df['ISS'] = (
        1 - (item_sparsity_df["u_id"]/item_rating_max)) * 100

    return sparsity, user_sparsity_df['USS'].mean(), item_sparsity_df['ISS'].mean(), len(users), len(channels)


def execute(df, random_sample: List[int], alpha_values: List[int] = [40], factor_options: List[int] = [150], iteration_opttions: List[int] = [25],   confidence_measures: List[MeasureOfConfidence] = [SimpleMeasureOfConfidence(), BinaryMeasureOfConfidence(), LogMeasureOfConfidence()], show_progress: bool = False):
    sparsity, USS, ISS, users, channels = calculateSparsity(df)
    evaluation_results = []

    for alpha_val in alpha_values:
        for confidence_measure in confidence_measures:
            c_ui = df['score'].astype(float).copy()
            c_ui.apply(confidence_measure.calculate, args=(alpha_val,))

            sparse_item_user = sparse.csr_matrix(
                (c_ui, (df['c_id'].astype(int), df['u_id'].astype(int))))
            # sparse_user_item = sparse.csr_matrix(( c_ui, (df['u_id'], df['c_id'])))
            for randomInt in random_sample:
                data_train, data_test = evaluation.train_test_split(
                    sparse_item_user, 0.25, randomInt)

                for factors in factor_options:
                    for iterations in iteration_opttions:
                        regularization = 0.1
                        models = []

                        models.append(implicit.als.AlternatingLeastSquares(
                            num_threads=4,  factors=factors, regularization=regularization, iterations=iterations))
                        models.append(implicit.cpu.bpr.BayesianPersonalizedRanking(
                            num_threads=4, factors=factors, iterations=iterations))
                        models.append(
                            implicit.cpu.lmf.LogisticMatrixFactorization(num_threads=4))
                        models.append(implicit.nearest_neighbours.ItemItemRecommender(
                            num_threads=4, K=20))

                        for model in models:
                            try:
                                model.fit(
                                    data_train, show_progress=show_progress)
                                ranking = implicit.evaluation.ranking_metrics_at_k(
                                    model, data_train, data_test, K=10, show_progress=show_progress)
                                evalResult = EvalResults(name=model.__class__, confidence=confidence_measure.name(
                                ), metrics=ranking, factors=factors, alpha=alpha_val, iterations=iterations)
                                evaluation_results.append(evalResult)
                            except BaseException as e:
                                print(
                                    f'Failed when processing {model} with {e}:{type(e)}')
                                continue

    df_evals = pd.DataFrame.from_records([[er.name.__doc__.split("\n\n")[0], er.metrics["auc"], er.metrics["precision"], er.factors, er.iterations, er.alpha, er.confidence, sparsity, USS, ISS, users, channels, er.metrics["ndcg"],
                                         er.metrics["map"]] for er in evaluation_results], columns=["name", 'AUC', 'precision', 'factors', 'iterations', 'alpha', 'confidence_measure', "sparsity", "USS", "ISS", "users", "channels", 'ndcg', 'map'])
    df_evals = df_evals.sort_values(
        ['AUC', 'precision', 'iterations', 'factors'], ascending=[False, False, True, True])

    return df_evals


def collaborative_filtering_cluster(user_ids, df, random_sample):
    cluster_df = df[df["user_id"].isin(user_ids)].copy()
    if(cluster_df.empty):
        return
    cluster_df["score"] = cluster_df["msg_count"]
    return execute(cluster_df[["user_id", "channel_id", "score", "u_id", "c_id"]], random_sample)


def normalize(series):
    return (series-series.min()) / \
        (series.max()-series.min())


def create_init_dataframe(data, min_users_per_channel=5, public_only=True):
    df = pd.DataFrame.from_records([vars(cm) for cm in data.channel_members])
    df["index"] = df["channel_id"] + "-" + df["user_id"]
    df.set_index('index', inplace=True)

    df_grouped_users = df.groupby(["channel_id"]).count()
    allowed_channels = df_grouped_users[df_grouped_users["user_id"]
                                        > min_users_per_channel].index.array

    public_channels = [c.channel_id for c in data.channels]

    df = df[df["channel_id"].isin(allowed_channels)]
    if(public_only):
        df = df[df["channel_id"].isin(public_channels)]

    df['u_id'] = df['user_id'].astype("category").cat.codes
    df['c_id'] = df['channel_id'].astype("category").cat.codes
    df["score"] = df["msg_count"].copy()
    return df
