import shutil
import tempfile
import zipfile
from pathlib import Path
import pandas as pd

from pycelonis.api import IBC

from .analyses import (
    o2c_otdp_analysis,
    o2c_ddp_analysis,
    ap_audit_sampling_analysis,
    p2p_root_cause_analysis,
    ap_dp_analysis,
)


class Demo:
    """A demo package for a use case.
    
    Parameters
    ----------
    name : str
        The name that the objects will get.
    datamodel_url : str
        A link to a datamodel backup. 
        Backups are generally in "Dropbox (Celonis)/Celonis Content Store/12_Machine Learning/04_ML_Demos"
    datamodel_folder : str
        The name of the datamodel backup folder.
    analysis : patlib.Path
        Path to ana analysis backup.
    """

    def __init__(self, name, datamodel_url, datamodel_folder, analysis):
        kwargs = locals()
        del kwargs["self"]
        for k, v in list(kwargs.items()):
            setattr(self, k, v)


o2c_otdp_demo = Demo(
    name="O2C On-Time Delivery Prediction Demo",
    datamodel_url="https://www.dropbox.com/sh/s9wkekmcsnzdvbi/AAB7SStYmKQPPqdq5BciDvjva?dl=1",
    datamodel_folder="Backup of Datamodel - O2C datamodel",
    analysis=o2c_otdp_analysis,
)

ap_dp_demo = Demo(
    name="AP Duplicate Payments Demo",
    datamodel_url="https://www.dropbox.com/sh/esogf1sctqzq7by/AABa8TT8_5KJHcYeaB64qRzfa?dl=1",
    datamodel_folder="Backup of Datamodel - AP Duplicate Payments",
    analysis=ap_dp_analysis,
)

o2c_ddp_demo = Demo(
    name="O2C Delivery Delay Prediction Demo",
    datamodel_url="https://www.dropbox.com/sh/s9wkekmcsnzdvbi/AAB7SStYmKQPPqdq5BciDvjva?dl=1",
    datamodel_folder="Backup of Datamodel - O2C datamodel",
    analysis=o2c_ddp_analysis,
)

ap_audit_sampling_demo = Demo(
    name="AP - Intelligent Audit Sampling Demo",
    datamodel_url="https://www.dropbox.com/sh/yfvjdt1lbmg12z1/AABOvVGxcdYjnvzoeXdp_J7Va?dl=1",
    datamodel_folder="Backup of Datamodel - AP datamodel",
    analysis=ap_audit_sampling_analysis,
)

p2p_root_cause_analysis_demo = Demo(
    name="P2P - Root Cause Analysis Demo",
    datamodel_url="https://www.dropbox.com/sh/8gvn6sosmxi31nx/AABoaY6TNON5KbPP9hDve7mFa?dl=1",
    datamodel_folder="Backup of Datamodel - P2P datamodel",
    analysis=p2p_root_cause_analysis,
)


def list_all_demos():
    return [o2c_ddp_demo]


def install_app(demo: Demo, celonis: IBC, target_workspace, analysis_name):
    """Create a Pool, Datamodel, Workspace and Analysis from demodata and backups."""
    a = target_workspace.create_analysis(analysis_name, demo.analysis)
    return a
