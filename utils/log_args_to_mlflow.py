#-------------------------------------------------------------------------------
# Name:        log_args_to_mlflow.py
# Purpose:     Log argparse Namespace to MLflow run parameters.
#-------------------------------------------------------------------------------

import mlflow


def log_args_to_mlflow(args):
    """
    Логирует все параметры командной строки в MLflow.
    Обрабатывает числа, строки, булевы значения и списки.
    """
    for arg_name, arg_value in vars(args).items():
        # Пропускаем аргументы, которые не нужно логировать (например, пути к данным)
        if arg_value is None:
            continue
        # Для списков (например, schedule) преобразуем в строку
        if isinstance(arg_value, list):
            arg_value = str(arg_value)
        mlflow.log_param(arg_name, arg_value)