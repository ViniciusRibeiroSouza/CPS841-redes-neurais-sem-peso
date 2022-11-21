import sys
import experiment
import wisardpkg as wp
from library import WizardParam


def main():
    # Number of splits used in KFold
    n_splits = 2

    # Parâmetros da WiSARD
    addressSize = None  # número de bits de enderaçamento das RAMs
    bleachingActivated = True  # desempate
    ignoreZero = False  # RAMs ignoram o endereço 0
    completeAddressing = True
    verbose = True  # mensanges durante a execução
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

    thresholds = [90, 100, 110, 120, 130]
    address_sizes = [3]
    for threshold in thresholds:
        for address_size in address_sizes:
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
