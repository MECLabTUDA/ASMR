# code from: https://github.com/LabSAINT/MUD-HoG_Federated_Learning/blob/main/server.py
from collections import Counter

import torch
import torch.nn.functional as F
import logging
from datetime import datetime, time
import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN


class HoGDetector:
    def __init__(self):
        pass

    def find_separate_point(self, d):
        # d should be flatten and np or list
        d = sorted(d)
        sep_point = 0
        max_gap = 0
        for i in range(len(d) - 1):
            if d[i + 1] - d[i] > max_gap:
                max_gap = d[i + 1] - d[i]
                sep_point = d[i] + max_gap / 2
        return sep_point

    def DBSCAN_cluster_minority(self, dict_data):
        ids = np.array(list(dict_data.keys()))
        values = np.array(list(dict_data.values()))
        if len(values.shape) == 1:
            values = values.reshape(-1, 1)
        cluster_ = DBSCAN(n_jobs=-1).fit(values)
        offset_ids = self.find_minority_id(cluster_)
        minor_id = ids[list(offset_ids)]
        return minor_id

    def find_minority_id(self, clf):
        count_1 = sum(clf.labels_ == 1)
        count_0 = sum(clf.labels_ == 0)
        mal_label = 0 if count_1 > count_0 else 1
        atk_id = np.where(clf.labels_ == mal_label)[0]
        atk_id = set(atk_id.reshape((-1)))
        return atk_id

    def find_majority_id(self, clf):
        counts = Counter(clf.labels_)
        major_label = max(counts, key=counts.get)
        major_id = np.where(clf.labels_ == major_label)[0]
        # major_id = set(major_id.reshape(-1))
        return major_id

    def mud_hog(self, clients):
        # long_HoGs for clustering targeted and untargeted attackers
        # and for calculating angle > 90 for flip-sign attack
        long_HoGs = {}

        # normalized_sHoGs for calculating angle > 90 for flip-sign attack
        normalized_sHoGs = {}
        full_norm_short_HoGs = []  # for scan flip-sign each round

        # L2 norm short HoGs are for detecting additive noise,
        # or Gaussian/random noise untargeted attack
        short_HoGs = {}

        # STAGE 1: Collect long and short HoGs.
        for i in range(self.num_clients):
            # longHoGs
            sum_hog_i = clients[i].get_sum_hog().detach().cpu().numpy()
            L2_sum_hog_i = clients[i].get_L2_sum_hog().detach().cpu().numpy()
            long_HoGs[i] = sum_hog_i

            # shortHoGs
            sHoG = clients[i].get_avg_grad().detach().cpu().numpy()
            # logging.debug(f"sHoG={sHoG.shape}") # model's total parameters, cifar=sHoG=(11191262,)
            L2_sHoG = np.linalg.norm(sHoG)
            full_norm_short_HoGs.append(sHoG / L2_sHoG)
            short_HoGs[i] = sHoG

            # Exclude the firmed malicious clients
            if i not in self.mal_ids:
                normalized_sHoGs[i] = sHoG / L2_sHoG

        # STAGE 2: Clustering and find malicious clients
        if self.iter >= self.tao_0:
            # STEP 1: Detect FLIP_SIGN gradient attackers
            """By using angle between normalized short HoGs to the median
            of normalized short HoGs among good candidates.
            NOTE: we tested finding flip-sign attack with longHoG, but it failed after long running.
            """
            flip_sign_id = set()
            """
            median_norm_shortHoG = np.median(np.array([v for v in normalized_sHoGs.values()]), axis=0)
            for i, v in enumerate(full_norm_short_HoGs):
                dot_prod = np.dot(median_norm_shortHoG, v)
                if dot_prod < 0: # angle > 90
                    flip_sign_id.add(i)
                    #logging.debug("Detect FLIP_SIGN client={}".format(i))
            logging.info(f"flip_sign_id={flip_sign_id}")
            """
            non_mal_sHoGs = dict(short_HoGs)  # deep copy dict
            for i in self.mal_ids:
                non_mal_sHoGs.pop(i)
            median_sHoG = np.median(np.array(list(non_mal_sHoGs.values())), axis=0)
            for i, v in short_HoGs.items():
                # logging.info(f"median_sHoG={median_sHoG}, v={v}")
                v = np.array(list(v))
                d_cos = np.dot(median_sHoG, v) / (np.linalg.norm(median_sHoG) * np.linalg.norm(v))
                if d_cos < 0:  # angle > 90
                    flip_sign_id.add(i)
                    # logging.debug("Detect FLIP_SIGN client={}".format(i))
            logging.info(f"flip_sign_id={flip_sign_id}")

            # STEP 2: Detect UNTARGETED ATTACK
            """ Exclude sign-flipping first, the remaining nodes include
            {NORMAL, ADDITIVE-NOISE, TARGETED and UNRELIABLE}
            we use DBSCAN to cluster them on raw gradients (raw short HoGs),
            the largest cluster is normal clients cluster (C_norm). For the remaining raw gradients,
            compute their Euclidean distance to the centroid (mean or median) of C_norm.
            Then find the bi-partition of these distances, the group of smaller distances correspond to
            unreliable, the other group correspond to additive-noise (Assumption: Additive-noise is fairly
            large (since it is attack) while unreliable's noise is fairly small).
            """

            # Step 2.1: excluding sign-flipping nodes from raw short HoGs:
            logging.info("===========using shortHoGs for detecting UNTARGETED ATTACK====")
            for i in range(self.num_clients):
                if i in flip_sign_id or i in self.flip_sign_ids:
                    short_HoGs.pop(i)
            id_sHoGs, value_sHoGs = np.array(list(short_HoGs.keys())), np.array(list(short_HoGs.values()))
            # Find eps for MNIST and CIFAR:
            """
            dist_1 = {}
            for k,v in short_HoGs.items():
                if k != 1:
                    dist_1[k] = np.linalg.norm(v - short_HoGs[1])
                    logging.info(f"Euclidean distance between 1 and {k} is {dist_1[k]}")
            logging.info(f"Average Euclidean distances between 1 and others {np.mean(list(dist_1.values()))}")
            logging.info(f"Median Euclidean distances between 1 and others {np.median(list(dist_1.values()))}")
            """

            # DBSCAN is mandatory success for this step, KMeans failed.
            # MNIST uses default eps=0.5, min_sample=5
            # CIFAR uses eps=50, min_sample=5 (based on heuristic evaluation Euclidean distance of grad of RestNet18.
            start_t = time.time()
            cluster_sh = DBSCAN(eps=self.dbscan_eps, n_jobs=-1,
                                min_samples=self.dbscan_min_samples).fit(value_sHoGs)
            t_dbscan = time.time() - start_t
            # logging.info(f"CLUSTER DBSCAN shortHoGs took {t_dbscan}[s]")
            # TODO: comment out this line
            logging.info("labels cluster_sh= {}".format(cluster_sh.labels_))
            offset_normal_ids = self.find_majority_id(cluster_sh)
            normal_ids = id_sHoGs[list(offset_normal_ids)]
            normal_sHoGs = value_sHoGs[list(offset_normal_ids)]
            normal_cent = np.median(normal_sHoGs, axis=0)
            logging.debug(f"offset_normal_ids={offset_normal_ids}, normal_ids={normal_ids}")

            # suspicious ids of untargeted attacks and unreliable or targeted attacks.
            offset_uAtk_ids = np.where(cluster_sh.labels_ == -1)[0]
            sus_uAtk_ids = id_sHoGs[list(offset_uAtk_ids)]
            logging.info(f"SUSPECTED UNTARGETED {sus_uAtk_ids}")

            # suspicious_ids consists both additive-noise, targeted and unreliable clients:
            suspicious_ids = [i for i in id_sHoGs if i not in normal_ids]  # this includes sus_uAtk_ids
            logging.debug(f"suspicious_ids={suspicious_ids}")
            d_normal_sus = {}  # distance from centroid of normal to suspicious clients.
            for sid in suspicious_ids:
                d_normal_sus[sid] = np.linalg.norm(short_HoGs[sid] - normal_cent)

            # could not find separate points only based on suspected untargeted attacks.
            # d_sus_uAtk_values = [d_normal_sus[i] for i in sus_uAtk_ids]
            # d_separate = find_separate_point(d_sus_uAtk_values)
            d_separate = self.find_separate_point(list(d_normal_sus.values()))
            logging.debug(f"d_normal_sus={d_normal_sus}, d_separate={d_separate}")
            sus_tAtk_uRel_id0, uAtk_id = set(), set()
            for k, v in d_normal_sus.items():
                if v > d_separate and k in sus_uAtk_ids:
                    uAtk_id.add(k)
                else:
                    sus_tAtk_uRel_id0.add(k)
            logging.info(f"This round UNTARGETED={uAtk_id}, sus_tAtk_uRel_id0={sus_tAtk_uRel_id0}")

            # STEP 3: Detect TARGETED ATTACK
            """
              - First excluding flip_sign and untargeted attack from.
              - Using KMeans (K=2) based on Euclidean distance of
                long_HoGs==> find minority ids.
            """
            for i in range(self.num_clients):
                if i in self.flip_sign_ids or i in flip_sign_id:
                    if i in long_HoGs:
                        long_HoGs.pop(i)
                if i in uAtk_id or i in self.uAtk_ids:
                    if i in long_HoGs:
                        long_HoGs.pop(i)

        out = self.FedFuncWholeNet(normal_clients, lambda arr: torch.mean(arr, dim=-1, keepdim=True))
        return out