"""
This module is an example of a barebones QWidget plugin for napari

It implements the Widget specification.
see: https://napari.org/stable/plugins/guides.html?#widgets

Replace code below according to your needs.
"""
import numpy as np
import pandas as pd
import tifffile
import matplotlib.pyplot as plt
import napari
import json

from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QFormLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QTabWidget,
    QWidget,
    QFileDialog
)
from qtpy.QtGui import QDoubleValidator
from pathlib import Path

from . import _utils as u


def is_Image_type(x):
    return isinstance(x, napari.layers.image.image.Image)


def is_Points_type(x):
    return isinstance(x, napari.layers.points.points.Points)


def is_Label_type(x):
    return isinstance(x, napari.layers.labels.labels.Labels)


def is_Tracks_type(x):
    return isinstance(x, napari.layers.tracks.tracks.Tracks)


class napariGEMspaWidget(QWidget):
    # your QWidget.__init__ can optionally request the napari viewer instance
    # in one of two ways:
    # 1. use a parameter called `napari_viewer`, as done here
    # 2. use a type annotation of 'napari.viewer.Viewer' for any parameter
    def __init__(self, napari_viewer):
        super().__init__()
        self.viewer = napari_viewer
        self._setup_ui()

        self.msds_data = {}
        self.pwdists_data = {}

        # if there are layers, update choices
        if len(self.viewer.layers) > 0:
            self._initialize_input_choices()

        # connect internal events whenever layers is changed
        self.viewer.layers.events.inserted.connect(self._update_on_inserted)
        self.viewer.layers.events.removed.connect(self._update_on_removed)

    def _setup_ui(self):
        # main layout
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)

        # tab widget
        main_tab = QTabWidget()

        # add main tab widget to main layout
        self.layout.addWidget(main_tab)

        # subwidget 1 : input & localization
        subwidget1 = QWidget()
        subwidget1_layout = QVBoxLayout()
        subwidget1.setLayout(subwidget1_layout)

        # input layers
        self.input_layer_combo = QComboBox(self)
        self.mask_layer_combo = QComboBox(self)

        # important image metadata
        self.pixel_size = QLineEdit()
        self._px_size_validator = QDoubleValidator()
        self.pixel_size.setValidator(self._px_size_validator)
        self.time_interval = QLineEdit()
        self._time_validator = QDoubleValidator()
        self.time_interval.setValidator(self._time_validator)

        # image parameters widget
        _image_params_widget = QWidget()
        _image_params_form = QFormLayout()
        _image_params_widget.setLayout(_image_params_form)

        # frame range
        self.frame_start = QSpinBox()
        self.frame_end = QSpinBox()
        self.frame_start.setRange(0, 1)
        self.frame_start.setValue(0)
        self.frame_end.setValue(1)
        self.frame_end.setRange(1, 1)

        _image_params_form.addRow("data", self.input_layer_combo)
        _image_params_form.addRow("mask", self.mask_layer_combo)
        _image_params_form.addRow("pixel size, um", self.pixel_size)
        _image_params_form.addRow("dt, s", self.time_interval)
        _image_params_form.addRow("frame start", self.frame_start)
        _image_params_form.addRow("frame end", self.frame_end)

        # localization parameters
        _localization_widget = QWidget()
        _localization_layout = QFormLayout()
        _localization_widget.setLayout(_localization_layout)

        self.laplace_sigma = QDoubleSpinBox()
        self.laplace_sigma.setRange(0.5, 10.0)
        self.laplace_sigma.setValue(1.5)
        self.laplace_sigma.setSingleStep(0.1)
        self.laplace_thres = QDoubleSpinBox()
        self.laplace_thres.setRange(0.0, 20000.0)
        self.laplace_thres.setValue(5.0)
        self.laplace_thres.setSingleStep(1.0)
        self.laplace_layername = QLineEdit("peaks")
        self.localization_button = QPushButton("Find spots")

        _localization_layout.addRow("sigma", self.laplace_sigma)
        _localization_layout.addRow("threshold", self.laplace_thres)
        _localization_layout.addRow("output name", self.laplace_layername)
        _localization_layout.addRow(None, self.localization_button)

        # add to first tab widget
        subwidget1_layout.addWidget(_image_params_widget)
        subwidget1_layout.addWidget(_localization_widget)
        main_tab.addTab(subwidget1, "Input/localization")

        # subwidget 2 : tracking / analysis
        subwidget2 = QWidget()
        subwidget2_layout = QVBoxLayout()
        subwidget2.setLayout(subwidget2_layout)

        _tracking_group = QGroupBox("Trajectory linking")
        _tracking_layout = QVBoxLayout()
        _tracking_group.setLayout(_tracking_layout)

        _tracking_subwidget = QWidget()
        _tracking_subwidget_layout = QFormLayout()
        _tracking_subwidget.setLayout(_tracking_subwidget_layout)

        self.laptrack_input_combo = QComboBox()
        self.laptrack_max_displacement = QDoubleSpinBox(decimals=4)

        # alter this when looking at different images via setMaximum()
        self.laptrack_max_displacement.setRange(0.0, 50.0)
        self.laptrack_max_displacement.setValue(1.0)
        self.laptrack_max_displacement.setSingleStep(0.1)

        _tracking_subwidget_layout.addRow(
            "max displacement, um", self.laptrack_max_displacement
        )

        self.tracking_button = QPushButton("link trajectories")
        _tracking_subwidget_layout.addRow(None, self.tracking_button)

        _tracking_layout.addWidget(self.laptrack_input_combo)
        _tracking_layout.addWidget(_tracking_subwidget)

        subwidget2_layout.addWidget(_tracking_group)

        # analysis parameters
        analysis_group = QGroupBox("Analysis")
        analysis_layout = QVBoxLayout()
        analysis_group.setLayout(analysis_layout)

        self.analysis_input_combo = QComboBox()
        self.plot_msd_button = QPushButton("Plot MSDs")
        self.D_init = QLineEdit()
        _D_validator = QDoubleValidator()
        self.D_init.setText("0.5")
        self.slope_init = QLineEdit()
        _slope_validator = QDoubleValidator()
        self.slope_init.setText("0.001")

        _fit_pars_widget = QWidget()
        _fit_pars_layout = QHBoxLayout()
        self.D_init.setValidator(_D_validator)
        self.slope_init.setValidator(_slope_validator)

        _fit_pars_widget.setLayout(_fit_pars_layout)
        _fit_pars_layout.addWidget(QLabel("D_init"))
        _fit_pars_layout.addWidget(self.D_init)
        _fit_pars_layout.addWidget(QLabel("slope_init"))
        _fit_pars_layout.addWidget(self.slope_init)

        self.plot_pwdists_button = QPushButton("Plot displacements")

        analysis_layout.addWidget(self.analysis_input_combo)
        analysis_layout.addWidget(self.plot_msd_button)
        analysis_layout.addWidget(_fit_pars_widget)
        analysis_layout.addWidget(self.plot_pwdists_button)
        analysis_group.setLayout(analysis_layout)

        subwidget2_layout.addWidget(analysis_group)

        # add subwidget 2 to main layout
        main_tab.addTab(subwidget2, "Tracking/analysis")

        self.save_state_button = QPushButton("Save state")
        self.load_state_button = QPushButton("Load state")

        self.layout.addWidget(self.save_state_button)
        self.layout.addWidget(self.load_state_button)
        self.layout.addStretch()

        # setup button behaviors
        self.input_layer_combo.currentTextChanged.connect(
            self._update_frame_range
        )
        self.localization_button.clicked.connect(
            self._find_spots_at_current_image
        )
        self.tracking_button.clicked.connect(
            self._link_trajectories_at_current_points
        )
        self.plot_msd_button.clicked.connect(self._plot_msd)
        self.plot_pwdists_button.clicked.connect(self._plot_pwdists_hist)

        self.save_state_button.clicked.connect(self.save_state)
        self.load_state_button.clicked.connect(self.load_state)

    def _update_on_inserted(self, event):
        # the layer that triggered the event:
        _layer = event.value
        _layer.events.name.connect(self.__update_on_renamed)
        # event.source seems to return the entire list of layers
        # comboboxes:
        # self.input_layer_combo (Image)
        # self.mask_layer_combo (Labels)
        # self.laptrack_input_combo (Points)
        if is_Image_type(_layer):
            self.input_layer_combo.addItem(_layer.name)
        elif is_Label_type(_layer):
            self.mask_layer_combo.addItem(_layer.name)
        elif is_Points_type(_layer):
            self.laptrack_input_combo.addItem(_layer.name)
        elif is_Tracks_type(_layer):
            self.analysis_input_combo.addItem(_layer.name)

    def _update_on_removed(self, event):
        _layer = event.value
        if is_Image_type(_layer):
            _layer_id = self.input_layer_combo.findText(_layer.name)
            self.input_layer_combo.removeItem(_layer_id)
        elif is_Label_type(_layer):
            _layer_id = self.mask_layer_combo.findText(_layer.name)
            self.mask_layer_combo.removeItem(_layer_id)
        elif is_Points_type(_layer):
            _layer_id = self.laptrack_input_combo.findText(_layer.name)
            self.laptrack_input_combo.removeItem(_layer_id)
        elif is_Tracks_type(_layer):
            _layer_id = self.analysis_input_combo.findText(_layer.name)
            self.analysis_input_combo.removeItem(_layer_id)

    def __update_on_renamed(self, name_event):
        _layer = name_event.source
        _new_name = _layer.name
        layer_names = set([layer.name for layer in self.viewer.layers])
        if is_Image_type(_layer):
            # figure out which layer has been renamed
            option_names = set(
                [
                    self.input_layer_combo.itemText(i)
                    for i in range(self.input_layer_combo.count())
                ]
            )
            _old_name = [x for x in option_names if x not in layer_names][0]
            _choice_index = self.input_layer_combo.findText(_old_name)
            self.input_layer_combo.setItemText(_choice_index, _new_name)
        if is_Label_type(_layer):
            # figure out which layer has been renamed
            option_names = set(
                [
                    self.mask_layer_combo.itemText(i)
                    for i in range(self.mask_layer_combo.count())
                ]
            )
            _old_name = [x for x in option_names if x not in layer_names][0]
            _choice_index = self.mask_layer_combo.findText(_old_name)
            self.mask_layer_combo.setItemText(_choice_index, _new_name)
        if is_Points_type(_layer):
            # figure out which layer has been renamed
            option_names = set(
                [
                    self.laptrack_input_combo.itemText(i)
                    for i in range(self.laptrack_input_combo.count())
                ]
            )
            _old_name = [x for x in option_names if x not in layer_names][0]
            _choice_index = self.laptrack_input_combo.findText(_old_name)
            self.laptrack_input_combo.setItemText(_choice_index, _new_name)

    def _initialize_input_choices(self):
        # initialize choices
        for layer in self.viewer.layers:
            layer.events.name.connect(self.__update_on_renamed)
            if is_Image_type(layer):
                self.input_layer_combo.addItem(layer.name)
            elif is_Label_type(layer):
                self.mask_layer_combo.addItem(layer.name)
            elif is_Points_type(layer):
                self.laptrack_input_combo.addItem(layer.name)
            elif is_Tracks_type(layer):
                self.analysis_input_combo.addItem(layer.name)

    def _update_frame_range(self, current_input):
        if current_input != "":
            # get current input frame range
            Nframes = self.viewer.layers[current_input].data.shape[0]
            # also get metadata
            metadata = self.viewer.layers[current_input].metadata

            try:
                dt_ms = (
                    metadata["aicsimage"]
                    .metadata["experiment"][0]
                    .parameters.periodDiff.avg
                )
            except:
                dt_ms = 0.0

            dy, dx = self.viewer.layers[current_input].scale[-2:]

            self.pixel_size.setText(f"{dy:0.4f}")
            self.time_interval.setText(f"{round(dt_ms)/1000.0:.3f}")

            self.frame_start.setRange(0, Nframes)
            self.frame_end.setRange(0, Nframes)
            self.frame_start.setValue(0)
            self.frame_end.setValue(min(10, Nframes))

    def _find_spots_at_current_image(self):
        current_layer = self.input_layer_combo.currentText()
        current_mask = self.mask_layer_combo.currentText()
        current_threshold = self.laplace_thres.value()
        current_sigma = self.laplace_sigma.value()
        frame_beg = self.frame_start.value()
        frame_end = self.frame_end.value()
        assert frame_end > frame_beg, "frame start must be BEFORE frame end!"

        target_layer_name = self.laplace_layername.text()

        # clip time axes for identifying spots
        data = self.viewer.layers[current_layer].data[
            frame_beg : frame_end + 1, :, :
        ]

        # assume that only the last two dimensions are displayed
        _dims_not_displayed = tuple(i for i in range(data.ndim - 2))

        if current_mask != "":
            #  ensure mask is 2D (whatever is displayed)
            mask = (
                self.viewer.layers[current_mask].data.max(
                    axis=_dims_not_displayed
                )
                > 0
            )
            # set uniform mask
            self.viewer.layers[current_mask].data[...] = mask
            self.viewer.layers[current_mask].refresh()
        else:
            mask = None

        xyloc_df = u.isolate_spots(
            data, mask=mask, threshold=current_threshold, sigma=current_sigma
        )

        # add physical units to x and y
        dxy = float(self.pixel_size.text())

        # add frame offset
        xyloc_df["frame"] += frame_beg

        # refine the spots against raw data
        u.refine_spots(self.viewer.layers[current_layer].data, xyloc_df)

        # create new columns with physical units
        xyloc_df["x_um"] = xyloc_df["x"].astype(float) * dxy
        xyloc_df["y_um"] = xyloc_df["y"].astype(float) * dxy

        # aicsimageio loads image with "xy" as the last axes...
        if target_layer_name not in self.viewer.layers:
            self.viewer.add_points(
                xyloc_df[["frame", "y_um", "x_um"]],
                name=target_layer_name,
                symbol="square",
                size=7 * dxy,
                edge_color="yellow",
                face_color="#ffffff00",
            )
        else:
            self.viewer.layers[target_layer_name].data = xyloc_df[
                ["frame", "y_um", "x_um"]
            ].to_numpy()
            # new points are automatically selected
            self.viewer.layers[target_layer_name].selected_data = set()
            # simply setting it to an empty set will deselect new points

    def _link_trajectories_at_current_points(self):
        current_layer = self.input_layer_combo.currentText()
        spacing = self.viewer.layers[current_layer].scale
        selected_points = self.laptrack_input_combo.currentText()
        data = self.viewer.layers[selected_points].data
        data_df = pd.DataFrame(data, columns=["frame", "y", "x"])
        # max dist
        max_disp = self.laptrack_max_displacement.value()

        tracks = u.link_spots_to_trajectory(data_df, max_displacement=max_disp)
        tracks_name = f"{selected_points}_tracks"

        if tracks_name not in self.viewer.layers:
            self.viewer.add_tracks(
                tracks[["track_id", "frame_y", "y", "x"]],
                tail_length=5,
                name=f"{selected_points}_tracks",
            )
        else:
            self.viewer.layers[tracks_name].data = tracks[
                ["track_id", "frame_y", "y", "x"]
            ]

    def _plot_msd(self):
        dt = float(self.time_interval.text())
        _current_track = self.analysis_input_combo.currentText()
        tracks = pd.DataFrame(
            self.viewer.layers[_current_track].data,
            columns=["track_id", "frame", "y", "x"],
        )

        # do ensemble average MSDs
        msds = (
            tracks.groupby("track_id").apply(u.compute_msd).reset_index(level=0)
        )
        avg_msd = (
            msds.groupby("lag")["MSD"]
            .agg(["mean", "std", "count"])
            .reset_index()
        )

        self.msds_data["data"] = avg_msd

        _x = avg_msd["lag"].to_numpy() * dt
        _y = avg_msd["mean"].to_numpy()
        _s = avg_msd["std"].to_numpy()
        slope, mse = u.fitline(_x, _y, _s)

        # slope / 2 * dimensionality => diffusion coefficient
        self.msds_data["D"] = slope / 4.0

        plt.ion()
        fig, ax = plt.subplots(figsize=(4, 3))
        ax.errorbar(_x, _y, yerr=_s, fmt="o", ecolor="gray")
        ax.plot(_x, slope * _x, "r--", lw=2)
        ax.set_xlabel("$\\tau$, seconds")
        ax.set_ylabel("MSD, $\mu m^2$")
        ax.set_title(f"{_current_track}\nD = {slope / 4.0:0.3f} $\mu m^2 / s$")
        plt.show()
        plt.ioff()

    def _plot_pwdists_hist(self):
        D_init = float(self.D_init.text())
        slope_init = float(self.slope_init.text())
        dt = float(self.time_interval.text())
        _current_track = self.analysis_input_combo.currentText()
        r_max = self.laptrack_max_displacement.value()
        tracks = pd.DataFrame(
            self.viewer.layers[_current_track].data,
            columns=["track_id", "frame", "y", "x"],
        )
        pwdists = u.get_paired_displacements(tracks)
        pwdists = pwdists[pwdists <= r_max]

        self.pwdists_data["data"] = pwdists

        mle_res = u.mle_fit(
            pwdists, init_D=D_init, init_b=slope_init, dt=dt, r_max=r_max
        )

        self.pwdists_data["D"] = mle_res['D']
        self.pwdists_data["background_slope"] = mle_res['slope']

        _dfine = np.linspace(0, r_max, num=200)

        plt.ion()
        fig, ax = plt.subplots(figsize=(4, 3))
        ax.hist(pwdists, 30, density=True, linewidth=1.25, edgecolor="k")
        ax.plot(
            _dfine,
            u.rayleigh_pdf(_dfine, mle_res["D"], mle_res["slope"], dt, r_max),
            "k-",
            lw=2,
        )
        ax.set_xlabel("distance, $\mu m$")
        ax.set_ylabel("density")
        ax.set_title(
            f"{_current_track}\n D={mle_res['D']:.3f} $\mu m^2 / s$, slope={mle_res['slope']:.4E}"
        )
        plt.show()
        plt.ioff()

    def save_state(self):
        current_image = self.input_layer_combo.currentText()
        current_image_path = Path(self.viewer.layers[current_image].source.path)
        parent_path = current_image_path.parent
        output_path = parent_path / current_image_path.stem
        if not output_path.exists():
            output_path.mkdir(exist_ok=True)

        # gather all masks
        masks_layers = [
            self.mask_layer_combo.itemText(i)
            for i in range(self.mask_layer_combo.count())
        ]

        tracks_layers = [
            self.analysis_input_combo.itemText(i)
            for i in range(self.analysis_input_combo.count())
        ]

        current_tracks = self.analysis_input_combo.currentText()

        # gather analysis parameters
        params = {
            "data": str(current_image_path),
            "dxy": self.pixel_size.text(),
            "dt": self.time_interval.text(),
            "frame_start": self.frame_start.value(),
            "frame_end": self.frame_end.value(),
            "spot_threshold": self.laplace_thres.text(),
            "spot_sigma": self.laplace_sigma.text(),
            "max_disp": self.laptrack_max_displacement.text(),
            "D_msd (um^2/s)": self.msds_data['D'],
            "D_pwd (um^2/s)": self.pwdists_data['D'],
            "bg_pwd": self.pwdists_data['background_slope'],
        }

        # save all layers data
        for mask in masks_layers:
            # save the max projection instead of entire timestacks
            _mask = self.viewer.layers[mask].data.max(axis=0).astype(np.uint8)

            tifffile.imwrite(
                output_path / f"{mask}_mask.tif",
                _mask
            )
            
        for track in tracks_layers:
            _track = self.viewer.layers[track].data
            np.savetxt(
                output_path / f"{track}_track.txt",
                _track,
            )

        # save to a JSON file
        with open(output_path / f"{current_tracks}_params.json", "w") as f:
            json.dump(params, f, indent=2)

        if self.msds_data is not None:
            self.msds_data["data"].to_csv(
                output_path / f"{current_tracks}_MSD.csv", index=False
            )

        if self.pwdists_data is not None:
            np.savetxt(
                output_path / f"{current_tracks}_pwdists.txt",
                self.pwdists_data["data"],
                fmt="%.5f"
            )

    def load_state(self):
        print("not implemented yet!")
        folder_path = Path(
            QFileDialog.getExistingDirectory(
            None, "Select a folder", "~/"
            )
        )
        print(folder_path)
        pass
