import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter, find_peaks
import warnings
from pathlib import Path, PosixPath


class Dataset:
    """Creates an object a dataset consisting of features x and targets y.

    :param x: Features
    :type x: ndarray
    :param y: Targets
    :type y: ndarray
    """

    def __init__(self, x, y):
        if not len(x) == len(y):
            raise InputError('Length of inputs x and y must be the same')

        self.x = x
        self.y = y

        self.smoothed = False
        self.maximum_found = False

        return

    def __call__(self):
        return self.x, self.y

    @staticmethod
    def savgol(y, window_length, polyorder=3):
        """Smooths data using Savitzky-Golay filtering.

        :param window: Number of data points to smooth over.
        :type window: int
        :param polyorder: Degree of smoothing polynomial. Defaults to 3.
        :type polyorder: int
        """

        if window_length % 2 == 0:
            warnings.warn(
                'Parameter window_length must be odd. Increased window by 1.')
            window += 1
        # Dummy in case y = self.y
        ynew = y

        if len(y.shape) > 1:
            # Smooth over multiple data
            for i, data in enumerate(y):
                ynew[i] = savgol_filter(data, window_length, polyorder)
        else:
            ynew = savgol_filter(y, window_length, polyorder)

        return ynew

    def find_maximum(self, y, **kwargs):
        """Finds the maximum of a y and the corresponding value for x.
        Can be used to find the maximum of loading curves.

        :param y: Data to find the maximum of.
        :type y: ndarray
        """

        if not isinstance(y, np.ndarray):
            y = np.asarray(y)

        if 'distance' in kwargs:
            distance = kwargs['distance']
        else:
            distance = 500

        if 'prominence' in kwargs:
            prominence = kwargs['prominence']
        else:
            prominence = 0.025

        if 'height' in kwargs:
            height = kwargs['height']
        else:
            height_flag = True

        if len(self.y.shape) > 1:
            ymax = np.zeros(len(y))
            for i in range(len(y)):
                if height_flag:
                    height = 0.5 * y[i, np.argmax(y[i])]
                argmax = find_peaks(y[i],
                                    distance=distance,
                                    height=height,
                                    prominence=prominence)[0][0]
                ymax[i] = y[i, argmax]
        else:
            if height_flag:
                height = 0.5 * y[np.argmax(y)]
            argmax = find_peaks(y,
                                distance=distance,
                                height=height,
                                prominence=prominence)[0][0]
            ymax = y[argmax]

        self.maximum_found = True

        return ymax

    def get_data(self):
        return self.x, self.y

    def get_ymax(self):
        return self.ymax

    def update_y(self, y):
        """Can be used to update self.y with the smoothed values.

        :param y: Target data
        :type y: ndarray
        """
        self.y = y

        return

    def smooth_y(self, y, smoothing='savgol', **kwargs):
        """Smooths data by a given smoothing method.

        :param smoothing: Defines the method for smoothing. Can be set to None.
                          Defaults to 'savgol' for Savitzky-Golay filtering.
        :type smoothing: str
        :param kwargs: Arguments for smoothing function. E.g. window and
                       polynomial order for Savitzky-Golay filter.

        """

        smoothers = {'savgol': self.savgol}

        window_length = kwargs['window_length']

        ysmooth = smoothers[smoothing](y=y, window_length=window_length)
        self.smoothed = True

        return ysmooth

    def prep_data_mlaq(self, window_length):
        """Smoothens y and finds the maximum values.
        """
        self.y = self.smooth_y(self.y, smoothing='savgol',
                               window_length=window_length)
        self.ymax = self.find_maximum(self.y)

    def save_data(self, path, y='ymax', method='numpy'):
        """Saves x and y to path.

        :param path: Path for saving files.
        :type path: str
        :param y: String wrapped in eval to specify which y to save to
                  file. Can be ymax or y. Defaults to ymax.
        :type y: str
        :param method: Which method to use for saving files. Defaults to numpy
                       using np.save(...).
        :type method: str or pathlib.PosixPath
        """
        if isinstance(path, str):
            path = Path(path)
        np.save(path / 'features', self.x)
        np.save(path / 'targets', self.eval(y))

    def extend_data(self, features=None, targets=None, ymax=None):
        """Extends x and y according to paths.

        :param features: Path to features or features. Defaults to None.
        :type features: pathlib.PosixPath or str or ndarray
        :param targets: Path to targets or targets. Defaults to None.
        :type targets: pathlib.PosixPath or str or ndarray
        """

        if features is not None:
            if isinstance(features, str) or isinstance(features, PosixPath):
                xnew = np.load(features)
            else:
                xnew = features

            self.x = np.concatenate((self.x, xnew), axis=0)

        if targets is not None:
            if isinstance(targets, str) or isinstance(targets, PosixPath):
                ynew = np.load(targets)
            else:
                ynew = targets

            if ymax is not None:
                self.ymax = np.concatenate((self.ymax, targets), axis=0)
            else:
                if self.smoothed:
                    ynew = self.smooth_y(ynew)
                if self.maximum_found:
                    ynew_max = self.find_maximum(ynew)

                self.y = np.concatenate((self.y, ynew), axis=0)
                if len(targets.shape[0]) > 1:
                    self.ymax = np.concatenate((self.ymax, ynew_max), axis=0)

    def extend_ymax(self, ymax_new):
        """Extends only ymax.
        """
        if isinstance(ymax_new, str) or isinstance(ymax_new, PosixPath):
            ynew = np.load(ymax_new)
        else:
            ynew = ymax_new

        self.ymax = np.concatenate((self.ymax, ynew_max), axis=0)

    def replace_y(self, new_y):
        """Replaces all of current y with new values.
        """
        self.y = new_y
