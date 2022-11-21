import sys
from multiprocessing import Process

import experiment
import wisardpkg as wp
from library import WizardParam


def main():
    # Subprocess
    parallel = True
    # Number of splits used in KFold
    n_splits = 2

    # Parâmetros da WiSARD
    addressSize = None  # número de bits de enderaçamento das RAMs
    bleachingActivated = True  # desempate
    ignoreZero = False  # RAMs
    completeAddressing = True
    verbose = True  # MSGs during execution
    returnActivationDegree = False
    returnConfidence = False
    returnClassesDegrees = False

    # Classes
    wizard_param = WizardParam()
    wizard_param.verbose = verbose
    wizard_param.ignoreZero = ignoreZero
    wizard_param.addressSize = addressSize
    wizard_param.returnConfidence = returnConfidence
    wizard_param.completeAddressing = completeAddressing
    wizard_param.bleachingActivated = bleachingActivated
    wizard_param.returnClassesDegrees = returnClassesDegrees
    wizard_param.returnActivationDegree = returnActivationDegree

    # Process
    procs = []
    # Todo: Work with more thresholds and sizes
    # thresholds X address_sizes
    thresholds = [item for item in range(15, 180, 25)]
    address_sizes = [item for item in range(3, 40, 5)]
    if parallel:
        for threshold in thresholds:
            for address_size in address_sizes:
                proc = Process(target=run_wizard, args=(address_size, n_splits, threshold, wizard_param))
                procs.append(proc)
                proc.start()
        # complete the processes
        for proc in procs:
            proc.join()
    else:
        for threshold in thresholds:
            for address_size in address_sizes:
                run_wizard(address_size, n_splits, threshold, wizard_param)


def run_wizard(address_size, n_splits, threshold, wizard_param):
    print("ADD: {0} -- THR: {1}".format(address_size, threshold))
    wizard_param.addressSize = address_size
    wizard_param.threshold = threshold
    wizard_param_dict = wizard_param.get_param()
    temp_param_dict = get_configs_dict(wizard_param_dict)
    wizard_model = wp.Wisard(wizard_param_dict['addressSize'], **temp_param_dict)
    experiment.run(wizard_model, wizard_param, n_splits=n_splits)


def get_configs_dict(wizard_param_dict):
    temp_param_dict = wizard_param_dict.copy()
    del temp_param_dict['threshold']
    del temp_param_dict['addressSize']
    return temp_param_dict


if __name__ == "__main__":
    sys.exit(main())
