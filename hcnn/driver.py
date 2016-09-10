"""Top-level routines, including:

* extract_features
* train_model
* find_best_model
* predict
* analyze
* fit_and_predict_one
* fit_and_predict_cross_validation
"""

import boltons.fileutils
import datetime
import glob
import json
import logging
import numpy as np
import os
import pandas as pd
import shutil
import sklearn.metrics

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


def get_slicer_from_feature(feature_mode):
    if 'wcqt' == feature_mode:
        slicer = streams.wcqt_slices
    elif 'hcqt' == feature_mode:
        slicer = streams.hcqt_slices
    else:
        slicer = streams.cqt_slices
    return slicer


class Driver(object):
    "Controller class for running experiments and holding state."

    @classmethod
    def available_experiments(cls, config_path):
        model_dir = os.path.expanduser(
            C.Config.load(config_path)['paths']['model_dir'])

        return [x for x in os.listdir(model_dir)
                if os.path.isdir(os.path.join(model_dir, x))]

    def __init__(self, config, partitions=None,
                 model_name=None,
                 experiment_name=None,
                 dataset=None, load_features=True,
                 skip_load_dataset=False,
                 skip_features=False,
                 skip_training=False,
                 skip_cleaning=False
                 ):
        """
        Parameters
        ----------
        config: str OR hcnn.config.Config
            Path to a config yaml file, or an instantiated config.
            The later is for testing; typical use case would be to load by
            string.

        partitions : str in ['rwc', 'philharmonia', 'uiowa']
            Selects which datsaet will be used as the test set.
            OR if the dataset given has a 'partitions' section in the config,
            uses the partition file located at the key given.

            Can only be None when extracting features; otherwise,
            this value is required.

        experiment_name : str or None
            Name of the experiment. This is used to
            name the files/parameters saved.

            It is required for many but not all functions.

        model_name : str or None
            Use this to specify the model configuration to use.
            Otherwise, tries to load it from the config.

        dataset : hcnn.data.dataset.Dataset or None
            If None, tries to use the config to load the default dataset,
            using the features specified in feature_mode.

            Otherwise, uses the data given. [Intended for testing!
            Use the config if possible.]

        load_features : bool
            If true, attempts to load the features files from the dataset.
        """
        if isinstance(config, str):
            self.config = C.Config.load(config)
        else:
            self.config = config

        self.experiment_name = experiment_name
        self.skip_features = skip_features
        self.skip_training = skip_training
        self.skip_cleaning = skip_cleaning

        # Initialize common paths 'n variables.
        self._init(model_name)
        if not skip_load_dataset:
            self.load_dataset(dataset=dataset, load_features=load_features)
            if partitions:
                self.setup_partitions(partitions)

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

    def _init(self, model_name):
        if model_name is not None:
            self.model_definition = model_name
        else:
            self.model_definition = self.config["model"]

        if self.model_definition:
            self.feature_mode = self.model_definition.split('_')[0]
        else:
            self.feature_mode = None

        self.max_files_per_class = self.config.get(
            "training/max_files_per_class", None)

        self.dataset = None

        if self.experiment_name:
            self._model_dir = os.path.join(
                os.path.expanduser(self.config["paths/model_dir"]),
                self.experiment_name)
            self._experiment_config_path = os.path.join(
                self._model_dir, self.config['experiment/config_path'])

            # if these don't exist, we're not actually running anything
            if self.model_definition and self.feature_mode:
                utils.create_directory(self._model_dir)

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
            logger.info("load_dataset() - Using dataset passed as a parameter")
            # If it's a str, it's a path.
            if isinstance(dataset, str):
                self.dataset = hcnn.data.dataset.Dataset.load(
                    dataset, data_root=self.data_root)
            elif isinstance(dataset, hcnn.data.dataset.Dataset):
                self.dataset = dataset
        else:
            logger.info(utils.colored(
                "load_dataset() - loading from {}".format(self.dataset_index)))
            self.dataset = hcnn.data.dataset.Dataset.load(
                self.dataset_index, data_root=self.data_root)
            logger.info(utils.colored("load_dataset() ... complete"))

        assert len(self.dataset) > 0

        # If we want the features, additionally add it to the dataset.
        if load_features and not self.skip_features:
            logger.info(utils.colored("load_dataset() - extracting features."))
            self.dataset = self.extract_features()
        elif self.skip_features:
            logger.info(utils.colored(
                "load_dataset() - skipping feature extraction on user flag."))

    def setup_partitions(self, test_partition):
        """Given the partition, setup the sets."""
        # If the dataset we have selected has partitions
        if 'partitions' in self.dataset_config:
            data_df = self.dataset.to_df()
            partition_file = self.dataset_config['partitions'][test_partition]
            # Load the parition_file
            self.partitions_df = pd.read_csv(partition_file, index_col=0)

            # set the train_set, valid_set, test_set from the original dataset
            # using the indexes from teh partition_file.
            self.train_set = hcnn.data.dataset.Dataset(
                data_df.loc[(
                    self.partitions_df['partition'] == 'train')])
            self.valid_set = hcnn.data.dataset.Dataset(
                data_df.loc[(
                    self.partitions_df['partition'] == 'valid')])
            self.test_set = hcnn.data.dataset.Dataset(
                data_df.loc[(
                    self.partitions_df['partition'] == 'test')])

            assert (len(self.train_set) + len(self.valid_set) +
                    len(self.test_set)) == len(self.dataset)
        else:
            raise ValueError(
                "partition files must be supplied for this dataset.")

        self._init_cross_validation(test_partition)

    def _init_cross_validation(self, test_set):
        self._cv_model_dir = os.path.join(self._model_dir, test_set)
        self._params_dir = os.path.join(
            self._cv_model_dir,
            self.config["experiment/params_dir"])
        self._training_loss_path = os.path.join(
            self._cv_model_dir, self.config['experiment/training_loss'])

        if os.path.exists(self._cv_model_dir):
            logger.warning("Cleaning old experiment: {}".format(
                self._cv_model_dir))
        utils.create_directory(self._cv_model_dir,
                               # aka if DO the clean, recreate.
                               recreate=(not self.skip_cleaning))
        utils.create_directory(self._params_dir)

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

    def train_model(self):
        """
        Train a model, writing intermediate params
        to disk.

        Trains for max_iterations or max_time, whichever is fewer.
        [Specified in the config.]
        """
        if self.skip_training:
            logger.info(utils.colored("--skip_training specified - skipping"))
            return True

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

        slicer = get_slicer_from_feature(self.feature_mode)

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
        train_stats = pd.DataFrame(columns=['timestamp',
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
        logger.info("Total iterations: {}".format(iter_count))
        logger.info("Trained for {}".format(timers.get("train")))
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
        return os.path.exists(self._training_loss_path)

    def find_best_model(self):
        """Perform model selection on the validation set with a binary search
        for minimum validation loss.

        (Bayesean optimization might be another approach?)

        Parameters
        ----------
        validation_df : pd.DataFrame
            Name of the held out dataset (used to specify the valid file)

        Returns
        -------
        results_df : pd.DataFrame
            DataFrame containing the resulting losses.
        """
        logger.info("Finding best model for {}".format(
            utils.colored(self.experiment_name, "magenta")))
        # if not self.check_features_input():
        #     logger.error("find_best_model features missing invalid.")
        #     return False

        validation_df = self.valid_set.to_df()

        # load all necessary config parameters from the ORIGINAL config
        original_config = C.Config.load(self._experiment_config_path)
        validation_error_file = os.path.join(
            self._cv_model_dir, original_config['experiment/validation_loss'])

        slicer = get_slicer_from_feature(self.feature_mode)
        t_len = original_config['training/t_len']

        if not os.path.exists(validation_error_file):
            model_files = glob.glob(
                os.path.join(self._params_dir, "params*.npz"))

            result_df, best_model = MS.CompleteLinearWeightedF1Search(
                model_files, validation_df, slicer, t_len,
                show_progress=True)()

            result_df.to_pickle(validation_error_file)
            best_path = os.path.join(self._params_dir,
                                     original_config['experiment/best_params'])
            shutil.copyfile(best_model['model_file'], best_path)
        else:
            logger.info("Model Search already done; printing previous results")
            result_df = pd.read_pickle(validation_error_file)
            # make sure model_iteration is an int so sorting makes sense.
            result_df["model_iteration"].apply(int)
            logger.info("\n{}".format(
                result_df.sort_values("model_iteration")))

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

        original_config = C.Config.load(self._experiment_config_path)
        params_file = os.path.join(self._params_dir,
                                   selected_param_file)
        slicer = get_slicer_from_feature(self.feature_mode)

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
        original_config = C.Config.load(self._experiment_config_path)
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

    def fit_and_predict_one(self, test_set, skip_training=False):
        """On a particular model, with a given set
        * train
        * model_selection
        * predict
        * analyze
        * Write all outputs to a file

        Parameters
        ----------
        skip_training : boolean
            For situations where you need to re-run model selection
            and prediction, skip up to model selection.

        Returns
        -------
        success : true if succeeded.
        """
        self.setup_partitions(test_set)
        logger.info("Beginning fit_and_predict_one:{}".format(test_set))
        result = False

        # Step 0: initialize the data for the current splits.
        # self.setup_data_splits(test_set)

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

    def fit_and_predict_cross_validation(self, skip_training=False):
        """Master loop for running cross validation across
        all datasets.

        Parameters
        ----------
        skip_training : boolean
            For situations where you need to re-run model selection
            and prediction, skip up to model selection.

        Returns
        -------
        success : bool
            True if succeeded end-to-end, False if anything failed.
        """
        logger.info("Beginning fit_and_predict_cross_validation")
        results = []
        for test_set in ["rwc", "uiowa", "philharmonia"]:
            results.append(self.fit_and_predict_one(test_set,
                                                    skip_training=False))
        final_result = all(results)
        logger.info("Completed fit_and_predict_cross_validation. Result={}"
                    .format(final_result))
        return final_result

    def validate_data(self):
        return True

    def collect_results(self, result_dir):
        """
        Moves the following files to result_dir/experiment_name:
            - [hold_out_set]/training_loss.pkl
            - [hold_out_set]/validation_loss.pkl
            - [hold_out_set]/model_[param_number]_predictions.pkl

        Parameters
        ----------
        result_dir : str
            The root destination results directory.
        """
        if not self.experiment_name:
            logger.error("No valid experiment_name; can't collect_results.")
            return False

        # Make sure result_dir/experiment name exists
        results_output_dir = os.path.join(result_dir, self.experiment_name)
        boltons.fileutils.mkdir_p(results_output_dir)

        # For each hold_out_set
        # TODO: find a master place - config file maybe? to make it
        #  so we stop re-writing these.

        experiment_results = {
            "experiment": self.experiment_name
        }
        for dataset in ['rwc', 'uiowa', 'philharmonia']:
            source_dir = os.path.join(self._model_dir, dataset)
            destination_dir = os.path.join(results_output_dir, dataset)
            boltons.fileutils.mkdir_p(destination_dir)

            training_loss_fn = self.config['experiment/training_loss']
            training_loss_source = os.path.join(source_dir, training_loss_fn)
            training_loss_dest = os.path.join(destination_dir,
                                              training_loss_fn)
            validation_loss_fn = self.config['experiment/validation_loss']
            validation_loss_source = os.path.join(source_dir, validation_loss_fn)
            validation_loss_dest = os.path.join(destination_dir,
                                                validation_loss_fn)

            # Copy the training and validation loss
            if os.path.isfile(training_loss_source):
                shutil.copyfile(training_loss_source, training_loss_dest)
            if os.path.isfile(validation_loss_source):
                shutil.copyfile(validation_loss_source, validation_loss_dest)

            # Now, the prediction file. But we have to make sure it
            # matches the format!
            prediction_glob = os.path.join(source_dir,
                                           "model_*_predictions.pkl")
            prediction_files = glob.glob(prediction_glob)
            prediction_file = (prediction_files[0]
                               if len(prediction_files) > 0 else None)
            if prediction_file:
                pred_destination = os.path.join(destination_dir,
                                                os.path.basename(prediction_file))
                prediction_df = pd.read_pickle(prediction_file).dropna()
                # To make this easy, we drop the nan's here.
                # Possibly this is going to bit me later.

                y_true = (prediction_df['y_true'] if 'y_true' in prediction_df
                          else prediction_df['target']).astype(np.int)
                y_pred = (prediction_df['y_pred'] if 'y_pred' in prediction_df
                          else prediction_df['vote']).astype(np.int)
                new_prediction_df = pd.concat([y_pred, y_true], axis=1,
                                              keys=['y_pred', 'y_true'])
                new_prediction_df.to_pickle(pred_destination)

                # These are to make sklearn.metrics happy.
                # y_true = y_true.tolist()
                # y_pred = y_pred.tolist()
                experiment_results[dataset] = {
                    'prediction_file': pred_destination,
                    'mean_accuracy': float(sklearn.metrics.accuracy_score(
                        y_true, y_pred)),
                    'mean_precision': float(sklearn.metrics.precision_score(
                        y_true, y_pred, average='macro')),  # weighted?
                    'mean_recall': float(sklearn.metrics.recall_score(
                        y_true, y_pred, average='macro')),  # weighted?
                    'mean_f1': float(sklearn.metrics.f1_score(
                        y_true, y_pred, average='macro')),  # weighted?
                    'class_precision': sklearn.metrics.precision_score(
                        y_true, y_pred, average=None).tolist(),  # weighted?
                    'class_recall': sklearn.metrics.recall_score(
                        y_true, y_pred, average=None).tolist(),  # weighted?
                    'class_f1': sklearn.metrics.f1_score(
                        y_true, y_pred, average=None).tolist(),  # weighted?
                    'sample_weight': np.array(
                        new_prediction_df['y_true'].value_counts()).tolist()
                }

        experiment_results_file = os.path.join(
            results_output_dir, "experiment_results.json")
        with open(experiment_results_file, 'w') as fh:
            json.dump(experiment_results, fh, indent=2)

        return True
