import logging
import numpy as np
import pandas
import progressbar

import wcqtlib.common.utils as utils
import wcqtlib.data.parse as parse
import wcqtlib.train.models
instrument_map = parse.InstrumentClassMap()

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
        """
        self.param_list = param_list
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
        model = wcqtlib.train.models.NetworkManager.deserialize_npz(
            params_file)
        # Results contains one point accross the whole dataset
        logger.debug("Evaluating model: {}".format(params_file))
        results = evaluate_dataframe(
            self.valid_df, model, self.slicer_fx, self.t_len,
            show_progress=self.show_progress)
        # Convert it to the mean over the whole dataframe.
        evaluation_results = results[['mean_loss', 'mean_acc']].mean(axis=0)
        # Include the metadata in the series.
        return evaluation_results.append(pandas.Series({
            "model_file": params_file,
            "model_iteration": utils.filebase(params_file)[6:]
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
        start_ind = 0
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


def evaluate_one(dfrecord, model, slicer_fx, t_len):
    """Return an evaluation object/dict after evaluating
    a single model using a loaded model.

    This method runs the model.predict over every set of
    frames generated by slicer_fx, and returns
    the class with the maximum

    Parameters
    ----------
    dfrecord : pandas.DataFrame
        pandas.Series containing the record to evaluate.

    model : models.NetworkManager

    slicer_fx : function
        Function that extracts featuers fr eaach frame
        from the feature file.

    t_len : int

    Returns
    ------
    results : pandas.Series
        All the results stored as a pandas.Series
    """
    # Get predictions for every timestep in the file.
    results = []
    losses = []
    accs = []
    target = instrument_map.get_index(dfrecord["instrument"])

    for frames in slicer_fx(dfrecord, t_len=t_len,
                            shuffle=False, auto_restart=False):
        results += [model.predict(frames)]
        loss, acc = model.evaluate(frames)
        losses += [loss]
        accs += [acc]
    results = np.concatenate(results)
    mean_loss = np.mean(losses)
    mean_acc = np.mean(accs)

    class_predictions = results.argmax(axis=1)

    # calculate the maximum likelyhood class - the class with the highest
    #  predicted probability across all frames.
    max_likelyhood_class = results.max(axis=0).argmax()

    # Also calculate the highest voted frame.
    vote_class = np.asarray(np.bincount(class_predictions).argmax(),
                            dtype=np.int)

    # Return both of these as a dataframe.
    return pandas.Series(
        data=[mean_loss, mean_acc, max_likelyhood_class, vote_class, target],
        index=["mean_loss", "mean_acc", "max_likelyhood", "vote", "target"],
        name=dfrecord.name)


def evaluate_dataframe(test_df, model, slicer_fx, t_len, show_progress=False):
    """Run evaluation on the files in a dataframe.

    Parameters
    ----------
    test_df : pandas.DataFrame
        DataFrame pointing to the features files and targets to evaluate.

    model : models.NetworkManager

    slicer_fx : function
        Function that extracts featuers fr eaach frame
        from the feature file.

    t_len : int

    Returns
    -------
    results_df : pandas.DataFrame
        DataFrame containing the results from for each file,
        where the index of the original file is maintained, but
        the dataframe now contains the columns:
            * max_likelyhood
            * vote
            * target
    """
    results = []
    if show_progress:
        i = 0
        progress = progressbar.ProgressBar(max_value=len(test_df))

    try:
        for index, row in test_df.iterrows():
            results += [evaluate_one(row, model, slicer_fx, t_len)]

            if show_progress:
                progress.update(i)
                i += 1
    except KeyboardInterrupt:
        logger.error("Evaluation process interrupted; {} of {} evaluated."
                     .format(len(results), len(test_df)))
        logger.error("Recommend you start this process over to evaluate "
                     "them all.")

    return pandas.DataFrame(results)
