import numpy as np
from models.loda_dumb import LODA
from sklearn.metrics import average_precision_score, roc_auc_score, recall_score, accuracy_score

class FedLODA:
    def __init__(self, n_random_cuts = 100, name="loda", **kwargs):
        self.models = []
        self.sizes = []
        self.loda_extra_args = kwargs
        self.fitted = False
        self.n_proj = n_random_cuts

    def fit(self, X_list, y_list = None, contams = None, eval_on_train=True):
        if y_list is None:
            if isinstance(contams, list):
                assert len(contams)==len(X_list)
                for i in contams:
                    assert 0.<contams[i]<1.
            elif isinstance(contams, float):
                assert 0.<contams<1.
        elif isinstance(y_list, list):
            assert len(X_list)==len(y_list)
            contams = [y.mean() for y in y_list]

        n_agents = len(X_list)
        assert len(set([i.shape[1] for i in X_list]))==1
        n_components = X_list[0].shape[1]

        self.main_loda = LODA(dens_estim_method = "histogram",
            n_random_cuts = self.n_proj,
            **self.loda_extra_args)
        proj_mat = self.main_loda._projection(n_components)
        self.main_loda.projections_ = proj_mat

        #########################################################################
        # federated learning of the histogram parameters
        #########################################################################

        all_min = np.zeros((n_agents, self.n_proj))
        all_max = np.zeros((n_agents, self.n_proj))
        all_tts = np.zeros((n_agents, self.n_proj))
        all_iqr = np.zeros((n_agents, self.n_proj))

        all_obs = 0
        for i, (X, c) in enumerate(zip(X_list, contams)):

            projected_data = proj_mat.dot(X.T).T

            if len(projected_data.shape) != 2:
                projected_data = projected_data.reshape(-1, 1)
                if len(projected_data.shape) != 2:
                    raise ValueError("Data must be of shape (obs, dims).")
            obs = projected_data.shape[0]
            # print(projected_data.shape)

            all_min[i,:] = np.min(projected_data, axis=0)
            all_max[i,:] = np.max(projected_data, axis=0)
            all_tts[i,:] = np.var(projected_data, ddof=0, axis=0) * obs
            all_obs += obs
            all_iqr[i,:] = (np.percentile(projected_data, q=75, axis=0) - np.percentile(projected_data, q=25, axis=0)) 

        fed_min = np.min(all_min, axis=0)
        fed_max = np.max(all_max, axis=0)
        fed_std = np.sqrt(np.sum(all_tts, axis=0) / all_obs)
        fed_std *= 3.49
        fed_iqr = np.median(all_iqr, axis=0)
        fed_iqr *=2
        fed_iqr[fed_iqr==0] = np.infty
        fed_std = np.minimum(fed_std, fed_iqr)
        fed_bw = fed_std * np.power(all_obs, -1.0 / 3.0)
        assert np.all(fed_bw>0)

        fed_n_bins = np.ceil((fed_max - fed_min) / fed_bw).astype(np.int32) + 1
        # print(fed_n_bins)

        assert len(fed_n_bins) == self.n_proj

        # self.main_loda.fit(np.zeros(1, n_components))

        self.main_loda.prmin_ = fed_min
        self.main_loda.prmax_ = fed_max
        self.main_loda.prbw_ = np.empty(self.n_proj)
        self.main_loda.hists_ = [np.zeros(i) for i in fed_n_bins]
        self.main_loda.prnbins_ = fed_n_bins
        self.main_loda.prbw_ = (fed_max - fed_min) / (fed_n_bins-1)
        self.main_loda.prmin_ -= self.main_loda.prbw_ / 2
        self.main_loda.prmax_ += self.main_loda.prbw_ / 2

        
        # print([i.shape for i in self.main_loda.hists_])

        #########################################################################
        # learning of histograms based on federated parameters
        #########################################################################

        for i, (X, c) in enumerate(zip(X_list, contams)):
            loda = LODA(contamination=c, 
                        dens_estim_method = "histogram",
                        n_bins = fed_n_bins,
                        n_random_cuts=self.n_proj,
                        **self.loda_extra_args)
            loda.fit(X, predef_projmat = proj_mat, 
                     predef_mins=fed_min, predef_maxs=fed_max)
                     
            # print([i.shape for i in loda.hists_])
            self.models.append(loda)
            self.sizes.append(X.shape[0])
            for j in range(self.n_proj):
                self.main_loda.hists_[j] += (loda.hists_[j] - 1e-12) * X.shape[0] 
        
        for j in range(self.n_proj):
            self.main_loda.hists_[j] /= all_obs
            self.main_loda.hists_[j] += 1e-12

        self.fed_contam = np.mean(contams)
        self.fitted = True
        self.proj_mat = proj_mat

        return self

    def decision_function(self, X):
        assert self.fitted
        # n_proj = self.n_proj
        # pred_scores = np.zeros(X.shape[0])
        # # print(pred_scores.shape)
        # for size, mod in zip(self.sizes, self.models):
        #     pred_unscaled = mod.decision_function(X) * n_proj + n_proj * np.log(size)
        #     # pred_unscaled = mod.decision_function(X) * size
        #     # print(pred_unscaled.shape)
        #     pred_scores += np.exp(pred_unscaled)
        # # pred_scores -= (n_proj * np.log(np.sum(self.sizes)))
        # pred_scores /= np.sum(self.sizes)**n_proj
        # pred_scores = np.log(pred_scores)
        # pred_scores /= n_proj
        # # pred_scores /= np.sum(self.sizes)

        pred_scores = self.main_loda.decision_function(X, check_fit=False)

        return pred_scores

    def valid_metrics(self, X, y_true, threshold=None):

        anoscores = self.decision_function(X)

        if threshold is None:
            threshold = np.percentile(anoscores, 100 * (1 - self.fed_contam))
        ano_labels = (anoscores > threshold).astype('int').ravel()

        auc = roc_auc_score(y_true, anoscores)
        ap = average_precision_score(y_true, anoscores)
        acc = accuracy_score(y_true, ano_labels)

        return {
            'accuracy': acc,
            'av_prec': ap,
            'auc': auc,
        }
        