import logging
import numpy as np
import pandas
import sklearn.metrics

import hcnn.common.utils as utils
import hcnn.train.models
from . import predict

logger = logging.getLogger(__name__)


class ModelSelector(object):
    """Class to choose a model given a list of model parameters."""
    def __init__(self, param_list, valid_df, slicer_fx, t_len,
                 show_progress=False):
        """
        Parameters
        ----------
        param_list : list of str
            List of paths to the nn params.

        valid_df : pandas.DataFrame
            Dataframe pointing to dataset files to use for evaluation.

        slicer_fx : function
            Function used to generate data from individual files.

        t_len : int
            Length of windows in time.

        show_progress : bool
            Print the progress during each evaluation step.

        percent_validation_set : float or None
            Percent (as a float) of the validation set to sample
            when finding the best model.
        """
        # The params list is generated by glob. It is NOT GUARANTEED
        #  to be in order. ... so we need to order it ourselves.
        param_map = {int(utils.iter_from_params_filepath(x)): x
                     for x in param_list}
        # Now create the param_list from the sorted keys
        self.param_list = [param_map[k] for k in sorted(param_map.keys())]
        self.valid_df = valid_df
        self.slicer_fx = slicer_fx
        self.t_len = t_len
        self.show_progress = show_progress

    def __call__(self):
        """Do the thing.

        Returns
        -------
        results : pandas.DataFrame
        selected_model : dict
            Containing with keys:
                model_file
                model_iteration
                mean_loss
        """
        return self.model_search()

    def model_search(self):
        """The search function. Linear / O(n) for the base class.

        Returns
        -------
        selected_model : dict
            Containing with keys:
                model_file
                model_iteration
                mean_loss
        """
        best_model = None
        results = []
        for i, model in enumerate(self.param_list):
            results += [self.evaluate_model(model)]
            model_choice = self.compare_models(best_model, results[-1])
            best_model = results[-1] if model_choice > 0 else best_model
        return pandas.DataFrame(results), best_model

    def compare_models(self, model_a, model_b):
        """Compare two models, return which one is best.

        Parameters
        ----------
        model_a : dict or None
        model_b : dict
            Dict containing the summary statistic to analyze.
            Uses the "mean_loss" key by default to do the analysis;
            subclasses can change this by overriding this function.

        Returns
        -------
        best_model : dict
            + if right is best, - if left is best
        """
        if model_a is None:
            return 1
        elif model_b is None:
            return -1
        else:
            return 1 if model_b['mean_loss'] < model_a['mean_loss'] else -1

    def evaluate_model(self, params_file):
        """Evaluate a model as defined by a params file, returning
        a single value (mean loss by default) to compare over the validation
        set."""
        model = hcnn.train.models.NetworkManager.deserialize_npz(
            params_file)
        # Results contains one point accross the whole dataset
        logger.debug("Evaluating params {}".format(params_file))
        validation_predictions_df = predict.predict_many(
            self.valid_df, model, self.slicer_fx, self.t_len,
            show_progress=True).dropna()

        evaluation_results = pandas.Series({
            "mean_loss": validation_predictions_df['loss'].mean(),
            "mean_acc": sklearn.metrics.accuracy_score(
                validation_predictions_df['y_true'].astype(np.int),
                validation_predictions_df['y_pred'].astype(np.int)),
            "f1_weighted": sklearn.metrics.f1_score(
                validation_predictions_df['y_true'].astype(np.int),
                validation_predictions_df['y_pred'].astype(np.int),
                average='weighted')
        })
        # Include the metadata in the series.
        model_iteration = utils.filebase(params_file)[6:]
        model_iteration = int(model_iteration) if model_iteration.isdigit() \
            else model_iteration

        return evaluation_results.append(pandas.Series({
            "model_file": params_file,
            "model_iteration": model_iteration
        }))


class BinarySearchModelSelector(ModelSelector):
    """Do model selection with binary search."""
    def model_search(self):
        """Do a model search with binary search.

        Returns
        -------
        results : pandas.DataFrame
        selected_model : dict or pandas.Series
            Containing with keys:
                model_file
                model_iteration
                mean_loss
                ...
        """
        results = {}
        # Don't allow the zero index; We should never select an
        #  untrained model!
        start_ind = 1 if len(self.param_list) > 0 else 0
        end_ind = len(self.param_list) - 1
        # start_ind = len(self.param_list)/2
        # end_ind = start_ind
        while start_ind != end_ind:
            logger.info("Model Search - L:{} R:{}".format(
                utils.filebase(self.param_list[start_ind]),
                utils.filebase(self.param_list[end_ind])))
            if start_ind not in results:
                model = self.param_list[start_ind]
                results[start_ind] = self.evaluate_model(model)
            if end_ind not in results:
                model = self.param_list[end_ind]
                results[end_ind] = self.evaluate_model(model)
            best_model = self.compare_models(
                results[start_ind], results[end_ind])

            new_ind = np.int(np.round((end_ind + start_ind) / 2))
            if (end_ind - start_ind) > 1:
                start_ind, end_ind = (new_ind, end_ind) if best_model >= 0 \
                    else (start_ind, new_ind)
            else:
                start_ind, end_ind = (new_ind, new_ind)

        logger.info("Selected model {} / {}".format(
            start_ind, self.param_list[start_ind]))
        return pandas.DataFrame.from_dict(results, orient='index'), \
            results[start_ind]

    def compare_models(self, model_a, model_b):
        """Overriden version from the parent class using the accuracy instead
        (since that seems to be a much better predictor of actually how
         our models are doing.)

        Parameters
        ----------
        model_a : dict or None
        model_b : dict
            Dict containing the summary statistic to analyze.
            Uses the "mean_loss" key by default to do the analysis;
            subclasses can change this by overriding this function.

        Returns
        -------
        best_model : dict
            + if right is best, - if left is best
        """
        if model_a is None:
            return 1
        elif model_b is None:
            return -1
        else:
            return 1 if model_b['mean_acc'] > model_a['mean_acc'] else -1


class CompleteLinearWeightedF1Search(ModelSelector):
    """
    """
    def model_search(self):
        """Do a model search with binary search.

        Returns
        -------
        results : pandas.DataFrame
        selected_model : dict or pandas.Series
            Containing with keys:
                model_file
                model_iteration
                mean_loss
                mean_acc
                f1_weighted
        """
        results = {}
        # Don't allow the zero index; We should never select an
        #  untrained model!
        index = 1 if len(self.param_list) > 0 else 0
        end_ind = len(self.param_list) - 1

        logger.info("Linear Model Search from:{} to:{} [total #: {}]".format(
            utils.filebase(self.param_list[index]),
            utils.filebase(self.param_list[end_ind]),
            len(self.param_list) - 1))

        # kinda hacky, but it'll do for now.
        increment_amount = int(np.round(min(max(10**(np.log10(
            len(self.param_list)) - 1), 1), 10)))

        while index < end_ind:
            logger.info("Evaluating {}".format(
                utils.filebase(self.param_list[index])))

            if index not in results:
                model = self.param_list[index]
                results[index] = self.evaluate_model(model)

            index += increment_amount

        results_df = pandas.DataFrame.from_dict(results, orient='index')
        selected_index = results_df['f1_weighted'].idxmax()

        # Now select the one with the lowest score
        logger.info(
            utils.colored("Selected model index:{} / params: {}".format(
                selected_index,
                utils.filebase(self.param_list[selected_index]))))
        logger.info("For reference, here's the model selection results:")
        logger.info("\n{}".format(results_df.to_string()))
        return results_df, results[selected_index]
