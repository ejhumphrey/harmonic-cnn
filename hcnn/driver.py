"""Top-level routines, including:

* extract_features
* train_model
* find_best_model
* predict
* analyze
* fit_and_predict_one
* fit_and_predict_cross_validation
"""

import datetime
import glob
import logging
import numpy as np
import os
import pandas
import shutil

import hcnn.common.config as C
import hcnn.common.utils as utils
import hcnn.data.cqt
import hcnn.data.dataset
import hcnn.train.models as models
import hcnn.train.streams as streams
import hcnn.evaluate.analyze
import hcnn.evaluate.model_selection as MS
import hcnn.evaluate.predict

logger = logging.getLogger(__name__)


class EarlyStoppingException(Exception):
    pass


class StaleFeaturesError(Exception):
    pass


class NoFeaturesException(Exception):
    pass


def get_slicer_from_network_def(network_def_name):
    if 'wcqt' in network_def_name:
        slicer = streams.wcqt_slices
    elif 'hcqt' in network_def_name:
        slicer = streams.hcqt_slices
    else:
        slicer = streams.cqt_slices
    return slicer


class Driver(object):
    "Controller class for handling state."

    def __init__(self, config, experiment_name=None,
                 dataset=None, load_features=True):
        """
        Parameters
        ----------
        config: hcnn.config.Config
            Instantiated config.

        experiment_name : str or None
            Name of the experiment. This is used to
            name the files/parameters saved.

            It is required for many but not all functions.

        load_features : bool
            L
        """
        self.config = config
        self.experiment_name = experiment_name

        # Initialize common paths 'n variables.
        self._init()

        self.load_dataset(dataset=dataset, load_features=load_features)

    @property
    def selected_dataset(self):
        return self.config['data/selected']

    @property
    def dataset_config(self):
        return self.config['data/{}'.format(self.selected_dataset)]

    @property
    def dataset_index(self):
        return self.dataset_config['notes_index']

    @property
    def data_root(self):
        return self.dataset_config['root']

    @property
    def feature_dir(self):
        return os.path.expanduser(
            self.config['paths/feature_dir'])

    @property
    def features_path(self):
        dataset_fn = os.path.basename(self.dataset_index)
        return os.path.join(self.feature_dir, dataset_fn)

    def _init(self):
        self.model_definition = self.config["model"]
        self.max_files_per_class = self.config.get(
            "training/max_files_per_class", None)

        self.dataset = None

        if self.experiment_name:
            self._model_dir = os.path.join(
                os.path.expanduser(self.config["paths/model_dir"]),
                self.experiment_name)
            self._experiment_config_path = os.path.join(
                self._model_dir, self.config['experiment/config_path'])

            utils.create_directory(self._model_dir)

    def _init_cross_validation(self, test_set):
        self._cv_model_dir = os.path.join(self._model_dir, test_set)
        self._params_dir = os.path.join(
            self._cv_model_dir,
            self.config["experiment/params_dir"])
        self._training_loss_path = os.path.join(
            self._cv_model_dir, self.config['experiment/training_loss'])

        self._train_set_save_path = os.path.join(
            self._cv_model_dir,
            self.config['experiment/data_split_format'].format(
                "train", test_set))
        self._valid_set_save_path = os.path.join(
            self._cv_model_dir,
            self.config['experiment/data_split_format'].format(
                "valid", test_set))

        utils.create_directory(self._cv_model_dir)
        utils.create_directory(self._params_dir)

    @property
    def param_format_str(self):
        # Lazy instantiation for param_format_str
        if not hasattr(self, "_param_format_str") or \
                self._param_format_str is None:
            # set up the param formatter.
            max_iterations = self.config['training/max_iterations']
            params_zero_pad = int(np.ceil(np.log10(max_iterations)))
            param_format_str = self.config['experiment/params_format']
            # insert the zero padding into the format string.
            self._param_format_str = param_format_str.format(params_zero_pad)

        return self._param_format_str

    def _format_params_fn(self, model_iter):
        "Convert the model iteration index into a params filename."
        return self.param_format_str.format(model_iter)

    def _format_predictions_fn(self, model_iter):
        return os.path.join(
            self._cv_model_dir,
            self.config.get('experiment/predictions_format', None).format(
                model_iter))

    def _format_analysis_fn(self, model_iter):
        return os.path.join(
            self._cv_model_dir,
            self.config.get('experiment/analysis_format', None).format(
                model_iter))

    def check_features_input(self):
        """Check to make sure everything's ready to run, including:
         * existing features
        """
        if "cqt" not in self.dataset.to_df().columns:
            logger.error("No features for input data; please extract first.")
            return False
        return True

    def load_dataset(self, dataset=None, load_features=True):
        """Load the selected dataset in specified in the config file.

        Parameters
        ----------
        load_features : bool
            If true, tries to load the features version of the dataset,
            else just loads the original specified version.
        """
        # Always start by loading the dataset.
        if dataset:
            # If it's a str, it's a path.
            if isinstance(dataset, str):
                self.dataset = hcnn.data.dataset.Dataset.load(
                    dataset, data_root=self.data_root)
            elif isinstance(dataset, hcnn.data.dataset.Dataset):
                self.dataset = dataset
        else:
            self.dataset = hcnn.data.dataset.Dataset.load(
                self.dataset_index, data_root=self.data_root)

        # If we want the features, additionally add it to the dataset.
        if load_features:
            self.dataset = self.extract_features()

    def print_stats(self):
        dataset_df = self.dataset.to_df()
        datasets = ["rwc", "uiowa", "philharmonia"]

        def print_datasetcount(dataset):
            print("{:<20} {:<30}".format(
                "{} count".format(dataset),
                len(dataset_df[dataset_df["dataset"] == dataset])))
        for dataset in datasets:
            print_datasetcount(dataset)

        def print_dataset_instcount(df, instrument):
            inst_filter = df[df["instrument"] == instrument]
            print("{:<20} {:<30} {:<30} {:<30}".format(
                "{} count".format(instrument),
                len(inst_filter[inst_filter["dataset"] == "rwc"]),
                len(inst_filter[inst_filter["dataset"] == "uiowa"]),
                len(inst_filter[inst_filter["dataset"] == "philharmonia"])))

        classmap = hcnn.common.labels.InstrumentClassMap()

        print("---------------------------")
        print("Datasets-Instrument count / dataset")
        print("---------------------------")
        print(utils.colored("{:<20} {:<30} {:<30} {:<30}".format(
            "item", "rwc", "uiowa", "philharmonia")))
        for inst in sorted(dataset_df["instrument"].unique()):
            if inst in classmap.allnames:
                print_dataset_instcount(dataset_df, inst)

    def extract_features(self):
        """Extract CQTs from all files collected in collect."""
        updated_ds = hcnn.data.cqt.cqt_from_dataset(
            self.dataset, self.feature_dir, **self.config["features/cqt"])

        if updated_ds is not None and \
                len(updated_ds) == len(self.dataset):
            write_path = os.path.join(
                self.feature_dir, os.path.basename(self.dataset_index))
            updated_ds.save(write_path)

        return updated_ds

    def _get_validation_splits(self, test_set):
        logger.info("[{}] Constructing training df".format(
            self.experiment_name))
        train_set, valid_set = self.dataset.train_valid_sets(
            test_set, max_files_per_class=self.max_files_per_class)
        logger.debug("[{}] training_df : {} rows".format(
            self.experiment_name, len(train_set)))
        # Save the dfs to disk so we can use them for validation later.
        train_set.save(self._train_set_save_path)
        valid_set.save(self._valid_set_save_path)

        return train_set, valid_set

    def setup_data_splits(self, test_set):
        # Important paths & things to load.
        self._init_cross_validation(test_set)
        if not self.check_features_input():
            logger.error("train_model setup state invalid.")
            return False

        # Set up the dataframes we're going to train with.
        self.train_set, self.valid_set = self._get_validation_splits(test_set)

    def train_model(self):
        """
        Train a model, writing intermediate params
        to disk.

        Trains for max_iterations or max_time, whichever is fewer.
        [Specified in the config.]
        """
        assert hasattr(self, 'train_set') and hasattr(self, 'valid_set')

        logger.info("Starting training for experiment: {}".format(
            self.experiment_name))

        # Save the config we used in the model directory, just in case.
        self.config.save(self._experiment_config_path)

        # Duration parameters
        max_iterations = self.config['training/max_iterations']
        max_time = self.config['training/max_time']  # in seconds

        # Collect various necessary parameters
        t_len = self.config['training/t_len']
        batch_size = self.config['training/batch_size']
        n_targets = self.config['training/n_targets']
        logger.debug("Hyperparams:\nt_len: {}\nbatch_size: {}\n"
                     "n_targets: {}\nmax_iterations: {}\nmax_time: {}s or {}h"
                     .format(t_len, batch_size, n_targets, max_iterations,
                             max_time, (max_time / 60. / 60.)))

        slicer = get_slicer_from_network_def(self.model_definition)

        # Set up our streamer
        logger.info("[{}] Setting up streamer".format(self.experiment_name))
        streamer = streams.InstrumentStreamer(
            self.train_set.to_df(), slicer, t_len=t_len, batch_size=batch_size)

        # create our model
        logger.info("[{}] Setting up model: {}".format(self.experiment_name,
                                                       self.model_definition))
        network_def = getattr(models, self.model_definition)(t_len, n_targets)
        model = models.NetworkManager(network_def)

        iter_print_freq = self.config.get(
            'training/iteration_print_frequency', None)
        iter_write_freq = self.config.get(
            'training/iteration_write_frequency', None)

        timers = utils.TimerHolder()
        iter_count = 0
        train_stats = pandas.DataFrame(columns=['timestamp',
                                                'batch_train_dur',
                                                'iteration', 'loss'])
        min_train_loss = np.inf

        timers.start("train")
        logger.info("[{}] Beginning training loop at {}".format(
            self.experiment_name, timers.get("train")))
        try:
            timers.start(("stream", iter_count))
            for batch in streamer:
                timers.end(("stream", iter_count))
                timers.start(("batch_train", iter_count))
                loss = model.train(batch)
                timers.end(("batch_train", iter_count))
                row = dict(timestamp=timers.get_end(
                            ("batch_train", iter_count)),
                           batch_train_dur=timers.get(
                            ("batch_train", iter_count)),
                           iteration=iter_count,
                           loss=loss)
                train_stats.loc[len(train_stats)] = row

                # Time Logging
                logger.debug("[Iter timing] iter: {} | loss: {} | "
                             "stream: {} | train: {}".format(
                                iter_count, loss,
                                timers.get(("stream", iter_count)),
                                timers.get(("batch_train", iter_count))))
                # Print status
                if iter_print_freq and (iter_count % iter_print_freq == 0):
                    mean_train_loss = \
                        train_stats["loss"][-iter_print_freq:].mean()
                    logger.info("Iteration: {} | Mean_Train_loss: {}"
                                .format(iter_count,
                                        utils.conditional_colored(
                                            mean_train_loss, min_train_loss)))
                    min_train_loss = min(mean_train_loss, min_train_loss)
                    # Print the mean times for the last n frames
                    logger.debug("Mean stream time: {}, Mean train time: {}"
                                 .format(
                                     timers.mean(
                                         "stream",
                                         iter_count - iter_print_freq,
                                         iter_count),
                                     timers.mean(
                                         "batch_train",
                                         iter_count - iter_print_freq,
                                         iter_count)))

                # save model, maybe
                if iter_write_freq and (iter_count % iter_write_freq == 0):
                    save_path = os.path.join(
                        self._params_dir,
                        self.param_format_str.format(iter_count))
                    logger.debug("Writing params to {}".format(save_path))
                    model.save(save_path)

                if datetime.datetime.now() > \
                        (timers.get("train") + datetime.timedelta(
                            seconds=max_time)):
                    raise EarlyStoppingException("Max Time reached")

                iter_count += 1
                timers.start(("stream", iter_count))
                # Stopping conditions
                if (iter_count >= max_iterations):
                    raise EarlyStoppingException("Max Iterations Reached")

        except KeyboardInterrupt:
            logger.warn(utils.colored("Training Cancelled", "red"))
            print("User cancelled training at epoch:", iter_count)
        except EarlyStoppingException as e:
            logger.warn(
                utils.colored("Training Stopped for {}".format(e), "red"))
            print("Training halted for: ", e)
        timers.end("train")

        # Print final training loss
        logger.info("Total iterations:".format(iter_count))
        logger.info("Trained for ".format(timers.get("train")))
        logger.info("Final training loss: {}".format(
            train_stats["loss"].iloc[-1]))

        # Make sure to save the final iteration's model.
        save_path = os.path.join(
            self._params_dir, self.param_format_str.format(iter_count))
        model.save(save_path)
        logger.info("Completed training for experiment: {}".format(
            self.experiment_name))

        # Save training loss
        logger.info("Writing training stats to {}".format(
            self._training_loss_path))
        train_stats.to_pickle(
            self._training_loss_path)

        # We need these files for models election, so make sure they exist
        return all([os.path.exists(self._training_loss_path),
                    os.path.exists(self._training_loss_path)])

    def find_best_model(self):
        """Perform model selection on the validation set with a binary search
        for minimum validation loss.

        (Bayesean optimization might be another approach?)

        Parameters
        ----------
        validation_df : pandas.DataFrame
            Name of the held out dataset (used to specify the valid file)

        Returns
        -------
        results_df : pandas.DataFrame
            DataFrame containing the resulting losses.
        """
        logger.info("Finding best model for {}".format(
            utils.colored(self.experiment_name, "magenta")))
        if not self.check_features_input():
            logger.error("find_best_model features missing invalid.")
            return False

        validation_df = pandas.read_pickle(self._valid_set_save_path)

        # load all necessary config parameters from the ORIGINAL config
        original_config = C.Config.from_yaml(self._experiment_config_path)
        validation_error_file = os.path.join(
            self._cv_model_dir, original_config['experiment/validation_loss'])

        slicer = get_slicer_from_network_def(original_config['model'])
        t_len = original_config['training/t_len']

        if not os.path.exists(validation_error_file):
            model_files = glob.glob(
                os.path.join(self._params_dir, "params*.npz"))

            result_df, best_model = MS.BinarySearchModelSelector(
                model_files, validation_df, slicer, t_len,
                show_progress=True)()

            result_df.to_pickle(validation_error_file)
            best_path = os.path.join(self._params_dir,
                                     original_config['experiment/best_params'])
            shutil.copyfile(best_model['model_file'], best_path)
        else:
            logger.info("Model Search already done; printing previous results")
            result_df = pandas.read_pickle(validation_error_file)
            # make sure model_iteration is an int so sorting makes sense.
            result_df["model_iteration"].apply(int)
            logger.info("\n{}".format(
                result_df.sort_values("model_iteration")))

        # if plot_loss:
        # Load the training loss
        # training_loss = pandas.read_pickle(self._training_loss_path)
        #     fig = plt.figure()
        #     ax = plt.plot(training_loss["iteration"], training_loss["loss"])
        #     ax_val = plt.plot(result_df["model_iteration"], result_df["mean_loss"])
        #     plt.draw()
        #     plt.show()

        return result_df

    def select_best_iteration(self, model_selection_df):
        """Given the model selection df, return the iteration
        which produced the best model.

        Returns
        -------
        best_model : int
            The iteration number which produced the best model.
        """
        best = model_selection_df.loc[model_selection_df["mean_acc"].argmax()]
        return best["model_iteration"]

    def predict(self, model_iter):
        """Generates a prediction for *all* files, and writes them to disk
        as a dataframe.

        If features_df_override, replace the features_df with this
        dataframe (for testing)
        """
        if not self.check_features_input():
            logger.error("predict - features missing.")
            return False

        logger.info("Evaluating experient {} with params from iter {}".format(
            utils.colored(self.experiment_name, "magenta"),
            utils.colored(model_iter, "cyan")))
        selected_param_file = self._format_params_fn(model_iter)

        original_config = C.Config.from_yaml(self._experiment_config_path)
        params_file = os.path.join(self._params_dir,
                                   selected_param_file)
        slicer = get_slicer_from_network_def(original_config['model'])

        logger.info("Deserializing Network & Params...")
        model = models.NetworkManager.deserialize_npz(params_file)

        dataset_df = self.dataset.to_df()
        logger.debug("Predicting across {} files.".format(
            len(dataset_df['cqt'].nonzero()[0])))

        predictions_df_path = self._format_predictions_fn(model_iter)

        t_len = original_config['training/t_len']
        logger.info("Running evaluation on all files...")
        predictions_df = hcnn.evaluate.predict.predict_many(
            dataset_df, model, slicer, t_len, show_progress=True)
        predictions_df.to_pickle(predictions_df_path)
        return predictions_df

    def analyze_from_predictions(self, model_iter, test_set):
        """Loads predictions from a file before calling analyze."""
        original_config = C.Config.from_yaml(self._experiment_config_path)
        analyzer = hcnn.evaluate.analyze.PredictionAnalyzer.from_config(
            original_config, self.experiment_name, model_iter, test_set)

        analysis_path = self._format_analysis_fn(model_iter)
        logger.info("Saving analysis to:".format(analysis_path))
        analyzer.save(analysis_path)

        return os.path.exists(analysis_path)

    def analyze(self, predictions, model_iter):
        logger.info("Evaluating experient {} with params from {}".format(
            utils.colored(self.experiment_name, "magenta"),
            utils.colored(model_iter, "cyan")))

        analyzer = hcnn.evaluate.analyze.PredictionAnalyzer(predictions)
        analysis_path = self._format_analysis_fn(model_iter)
        logger.info("Saving analysis to:".format(analysis_path))
        analyzer.save(analysis_path)

        return os.path.exists(analysis_path)

    def fit_and_predict_one(self, test_set):
        """On a particular model, with a given set
        * train
        * model_selection
        * predict
        * analyze
        * Write all outputs to a file

        Returns
        -------
        success : true if succeeded.
        """
        logger.info("Beginning fit_and_predict_one:{}".format(test_set))
        result = False

        # Step 0: initialize the data for the current splits.
        self.setup_data_splits(test_set)

        # Step 1: train
        result = self.train_model()

        # Step 2: model selection
        if result:
            results_df = self.find_best_model()
            best_iter = self.select_best_iteration(results_df)

            # Step 3: predictions
            predictions = self.predict(best_iter)

            # Step 4: analysis
            if result:
                self.analyze(predictions, best_iter)
            else:
                logger.error("Problem predicting on {}".format(test_set))
        else:
            logger.error("Problem with training on {}".format(test_set))

        logger.info("Completed fit_and_predict_one:{}. Result={}"
                    .format(test_set, result))
        return result

    def fit_and_predict_cross_validation(self):
        """Master loop for running cross validation across
        all datasets.

        Returns
        -------
        success : bool
            True if succeeded end-to-end, False if anything failed.
        """
        logger.info("Beginning fit_and_predict_cross_validation")
        results = []
        for test_set in ["rwc", "uiowa", "philharmonia"]:
            results.append(self.fit_and_predict_one(test_set))
        final_result = all(results)
        logger.info("Completed fit_and_predict_cross_validation. Result={}"
                    .format(final_result))
        return final_result
