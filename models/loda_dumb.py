# -*- coding: utf-8 -*-
"""
Loda: Lightweight on-line detector of anomalies
Based on the implementation of the package pyod.
"""
# Author: Mate Baranyi <baranyim@math.bme.hu>

import numpy as np
from sklearn.utils.validation import check_is_fitted
from sklearn.utils import check_array
from scipy import special 

from sklearn.preprocessing import MinMaxScaler, StandardScaler, Normalizer
from sklearn.random_projection import SparseRandomProjection
from sklearn.metrics import average_precision_score, roc_auc_score, recall_score, accuracy_score


class LODA:

    """
    Loda: Lightweight on-line detector of anomalies. See
    :cite:`pevny2016loda` for more information.
    """

    def __init__(self, n_bins=-1, n_random_cuts=100,
                 bw=None,
                 normalize_projections=False,
                 n_nonzeros=None,
                 dens_estim_method="histogram",
                 contamination=0.1,
                 standardize=True,
                 rand_proj_method="article"):

        if not (0. < contamination <= 0.5):
            raise ValueError("contamination must be in (0, 0.5], "
                             "got: %f" % contamination)
        self.contamination = contamination
        
        self.n_bins = n_bins
        self.bw = bw
        self.n_random_cuts = n_random_cuts
        self.weights = np.ones(n_random_cuts, dtype=np.float)
        self.norm_proj = normalize_projections
        self.n_nonzeros = n_nonzeros
        self.dens_meth = dens_estim_method
        self.standardize = standardize
        self.proj_meth = rand_proj_method

    def fit(self, X, y=None, seed=None, eval_on_train=True, predef_projmat = None,
            predef_mins=None, predef_maxs=None):
        """Fit detector. y is ignored in unsupervised methods.

        Parameters
        ----------
        X : numpy array of shape (n_samples, n_features)
            The input samples.

        y : Ignored
            Not used, present for API consistency by convention.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        if isinstance(seed, int):
            np.random.seed(seed)

        # validate inputs X and y (optional)
        X = check_array(X).copy()
        if self.standardize:
            skaler = StandardScaler()
            skaler.fit(X)
            X = skaler.transform(X)
            self.scaler = skaler

        # self._set_n_classes(y)
        pred_scores = np.zeros([X.shape[0], 1])

        n_components = X.shape[1]

        ##########################################################################
        # different projections
        ##########################################################################

        if predef_projmat is None:
            proj_mat = self._projection(n_components)
        else:
            proj_mat = predef_projmat

        projected_data = proj_mat.dot(X.T)  # n_random_cuts x n
        self.projections_ = proj_mat

        assert self.projections_.shape == (self.n_random_cuts, n_components)

        ##########################################################################
        # histogram estimates
        ##########################################################################

        if self.dens_meth == "hist_old":
            if self.n_bins is None:
                self.n_bins = int(2 * (X.shape[0] ** (1/3.)))+1  # Rice rule
                # print(self.n_bins)

            self.histograms_ = np.zeros((self.n_random_cuts, self.n_bins))
            self.limits_ = np.zeros((self.n_random_cuts, self.n_bins + 1))

            for i in range(self.n_random_cuts):
                self.histograms_[i, :], self.limits_[i, :] = np.histogram(
                    projected_data[i, :], bins=self.n_bins, density=False)
                self.histograms_[i, :] += 1e-12
                self.histograms_[i, :] /= np.sum(self.histograms_[i, :])
            
            halfwidth = (self.limits_[:, 1:2] - self.limits_[:, 0:1])/2
            self.limits_ = np.c_[self.limits_[:, 0:1] - halfwidth, self.limits_, self.limits_[:, -1:] + halfwidth]
            outofrange_values = 1e-15*np.ones((self.n_random_cuts, 1))
            self.histograms_ = np.c_[outofrange_values, self.histograms_, outofrange_values]

        elif self.dens_meth == "histogram":
            if predef_mins is None:
                self.prmin_ = np.min(projected_data.T, axis=0)
            else:
                assert len(predef_mins) == self.n_random_cuts
                self.prmin_ = predef_mins
            
            if predef_maxs is None:
                self.prmax_ = np.max(projected_data.T, axis=0)
            else:
                assert len(predef_maxs) == self.n_random_cuts
                self.prmax_ = predef_maxs

            self.prbw_ = np.empty(self.n_random_cuts)
            self.hists_ = []

            if self.n_bins is None:
                self.n_bins = int(2 * (X.shape[0] ** (1/3.))) +1  # Rice rule
                self.prnbins_ = np.ones(self.n_random_cuts) * self.n_bins
                # print(self.n_bins)
            elif isinstance(self.n_bins, int):
                if self.n_bins > 0:
                    self.prnbins_ = np.ones(self.n_random_cuts) * self.n_bins
                if self.n_bins == -1:
                    self.prbw_ = self.scotts_rule(projected_data.T)
                    self.prnbins_ = np.ceil((self.prmax_ - self.prmin_) / self.prbw_).astype(np.int32) +1
            elif isinstance(self.n_bins, (list, np.ndarray)):
                assert len(self.n_bins) == self.n_random_cuts
                self.prnbins_ = np.asarray(self.n_bins)

            self.prbw_ = (self.prmax_ - self.prmin_) / (self.prnbins_-1)
            self.prmin_ -= self.prbw_ / 2
            self.prmax_ += self.prbw_ / 2

            # print(self.prnbins_)

            for i in range(self.n_random_cuts):
                prd = projected_data[i, :]
                bins = np.floor((prd - self.prmin_[i]) / self.prbw_[i]).astype(np.int32) 
                # print(bins)
                bins[bins == self.prnbins_[i]] = self.prnbins_[i] - 1
                # try:
                self.hists_.append(np.bincount(bins, minlength=self.prnbins_[i]) / X.shape[0] + 1e-12)
                # print(np.bincount(bins).shape)
                # except ValueError:
                #     print(bins)

        ##########################################################################
        # calculate the scores for the training samples
        ##########################################################################

        if eval_on_train:
            pred_scores = self._decision_func_inner(projected_data)
            self.decision_scores_ = pred_scores
            self._process_decision_scores()

        return self

    def _projection(self, n_components):

        # n_components = X.shape[1]

        if self.proj_meth == "theory":
            desired_size = n_components * self.n_random_cuts
            proj_mat = np.zeros((n_components-1) * self.n_random_cuts)
            if callable(self.n_nonzeros):
                num_nonzeros = int(self.n_nonzeros(desired_size))
            elif isinstance(self.n_nonzeros, float):
                num_nonzeros = int(desired_size * self.n_nonzeros)
            elif self.n_nonzeros is None:
                num_nonzeros = int(np.sqrt(desired_size))

            num_nonzeros = max(num_nonzeros, self.n_random_cuts) - self.n_random_cuts
            proj_mat[:num_nonzeros]=1
            np.random.shuffle(proj_mat)
            proj_mat = np.reshape(proj_mat, (self.n_random_cuts, n_components-1))
            proj_mat = np.c_[proj_mat, np.ones(self.n_random_cuts)]
            [np.random.shuffle(x) for x in proj_mat]

            proj_mat *= np.random.randn(self.n_random_cuts, n_components)

            if self.norm_proj:
                proj_mat = Normalizer().fit_transform(proj_mat)

        elif self.proj_meth == "article":
            if callable(self.n_nonzeros):
                n_nonzero_components = int(self.n_nonzeros(n_components))+1
            elif isinstance(self.n_nonzeros, float):
                n_nonzero_components = int(self.n_nonzeros * n_components)+1
            elif self.n_nonzeros is None:
                n_nonzero_components = int(np.sqrt(n_components))+1  # article default
            else:
                n_nonzero_components = n_components

            n_zero_components = n_components - n_nonzero_components

            proj_mat = np.random.randn(
                    self.n_random_cuts, n_components)

            for i in range(self.n_random_cuts):
                rands = np.random.permutation(n_components)[:n_zero_components]
                proj_mat[i, rands] = 0.

            if self.norm_proj:
                proj_mat = Normalizer().fit_transform(proj_mat)
        
        return proj_mat


    def decision_function(self, X, check_fit = True):
        """Predict raw anomaly score of X using the fitted detector.

        The anomaly score of an input sample is computed based on different
        detector algorithms. For consistency, outliers are assigned with
        larger anomaly scores.

        Parameters
        ----------
        X : numpy array of shape (n_samples, n_features)
            The training input samples. Sparse matrices are accepted only
            if they are supported by the base estimator.

        Returns
        -------
        anomaly_scores : numpy array of shape (n_samples,)
            The anomaly score of the input samples.
        """
        if check_fit:
            check_is_fitted(self, ['projections_', 'decision_scores_',
                                'threshold_', 'labels_'])

        X = check_array(X).copy()
        if self.standardize:
            X = self.scaler.transform(X)
        # moving the projection out of the loop
        projected_data = self.projections_.dot(X.T)

        pred_scores = self._decision_func_inner(projected_data)

        return pred_scores

    def _decision_func_inner(self, projected_data):

        pred_scores = np.zeros([projected_data.shape[1], 1])  # n * 1

        if self.dens_meth == "hist_old":
            for i in range(self.n_random_cuts):
                # projected_data = self.projections_[i, :].dot(X.T)
                inds = np.searchsorted(self.limits_[i, :self.n_bins - 1],
                                       projected_data[i, :], side='left')
                pred_scores[:, 0] += -self.weights[i] * \
                    np.log(self.histograms_[i, inds])

        if self.dens_meth == "histogram":
            extr = -np.log(1e-15)
            for i in range(self.n_random_cuts):
                prd = projected_data[i, :]
                bins = np.floor((prd - self.prmin_[i]) / self.prbw_[i]).astype(np.int32)
                bins[np.abs(prd - self.prmax_[i]) <= 1e-15] -= 1
                inrange = (bins>=0) & (bins<self.prnbins_[i])    
                pred_scores[inrange, 0] += -np.log(self.hists_[i][bins[inrange]])
                pred_scores[~inrange, 0] += extr


        pred_scores = pred_scores.ravel()
        # Todo: why is this scaled by the n_cuts AGAIN?
        pred_scores /= self.n_random_cuts
        return pred_scores

    def scotts_rule(self, data, indep=True):
        """
        Scotts rule.

        Scott (1992, page 152)
        Scott, D.W. (1992) Multivariate Density Estimation. Theory, Practice and
        Visualization. New York: Wiley.

        Examples
        --------
        >>> data = np.arange(9).reshape(-1, 1)
        >>> ans = scotts_rule(data)
        >>> assert np.allclose(ans, 1.76474568962182)
        """
        if len(data.shape) != 2:
            data = data.reshape(-1, 1)
            if len(data.shape) != 2:
                raise ValueError("Data must be of shape (obs, dims).")

        obs, dims = data.shape
        if indep:
            dims = 1

        sigma = np.std(data, ddof=dims, axis=0)
        IQR = (np.percentile(data, q=75, axis=0) - np.percentile(data, q=25, axis=0)) 
        
        if self.dens_meth == "histogram":
            IQR *= 2
            IQR[IQR==0] = np.infty

            sigma *= 3.49
            sigma = np.minimum(sigma, IQR)
            
            scott_bw = sigma * np.power(obs, -1.0 / 3.0)

        assert np.all(scott_bw>0)
        return scott_bw


    # from here functions are just copied from the BaseDetector class of pyod
    def _process_decision_scores(self):
        """Internal function to calculate key attributes:

        - threshold_: used to decide the binary label
        - labels_: binary labels of training data

        Returns
        -------
        self
        """

        self.threshold_ = np.percentile(self.decision_scores_,
                                     100 * (1 - self.contamination))
        self.labels_ = (self.decision_scores_ > self.threshold_).astype('int').ravel()

        # calculate for predict_proba()

        self._mu = np.mean(self.decision_scores_)
        self._sigma = np.std(self.decision_scores_)

        return self

    def predict(self, X):
        """Predict if a particular sample is an outlier or not.

        Parameters
        ----------
        X : numpy array of shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        outlier_labels : numpy array of shape (n_samples,)
            For each observation, tells whether or not
            it should be considered as an outlier according to the
            fitted model. 0 stands for inliers and 1 for outliers.
        """

        check_is_fitted(self, ['decision_scores_', 'threshold_', 'labels_'])

        pred_score = self.decision_function(X)
        pred_labels = (pred_score > self.threshold_).astype('int').ravel()
        return pred_labels

    def predict_proba(self, X, method='linear'):
        """Predict the probability of a sample being outlier. Two approaches
        are possible:

        1. simply use Min-max conversion to linearly transform the outlier
           scores into the range of [0,1]. The model must be
           fitted first.
        2. use unifying scores, see :cite:`kriegel2011interpreting`.

        Parameters
        ----------
        X : numpy array of shape (n_samples, n_features)
            The input samples.

        method : str, optional (default='linear')
            probability conversion method. It must be one of
            'linear' or 'unify'.

        Returns
        -------
        outlier_probability : numpy array of shape (n_samples,)
            For each observation, tells whether or not
            it should be considered as an outlier according to the
            fitted model. Return the outlier probability, ranging
            in [0,1].
        """

        check_is_fitted(self, ['decision_scores_', 'threshold_', 'labels_'])
        train_scores = self.decision_scores_

        test_scores = self.decision_function(X)

        probs = np.zeros([X.shape[0], 2])
        if method == 'linear':
            scaler = MinMaxScaler().fit(train_scores.reshape(-1, 1))
            probs[:, 1] = scaler.transform(
                test_scores.reshape(-1, 1)).ravel().clip(0, 1)
            probs[:, 0] = 1 - probs[:, 1]
            return probs

        elif method == 'unify':
            # turn output into probability
            pre_erf_score = (test_scores - self._mu) / (
                    self._sigma * np.sqrt(2))
            erf_score = special.erf(pre_erf_score)
            probs[:, 1] = erf_score.clip(0, 1).ravel()
            probs[:, 0] = 1 - probs[:, 1]
            return probs
        else:
            raise ValueError(method,
                             'is not a valid probability conversion method')

    def valid_metrics(self, X, y_true, threshold=None):

        anoscores = self.decision_function(X)

        if threshold is None:
            threshold = self.threshold_
        ano_labels = (anoscores > threshold).astype('int').ravel()
        # print(y_true.shape, anoscores.shape)
        auc = roc_auc_score(y_true, anoscores)
        ap = average_precision_score(y_true, anoscores)
        acc = accuracy_score(y_true, ano_labels)

        return {
            'accuracy': acc,
            'av_prec': ap,
            'auc': auc,
        }
