import os
import typing
import shutil

import argparse

import pandas as pd
import numpy as np


parser = argparse.ArgumentParser(
    prog="CsvDatasetManager",
    description="split and save",
)
parser.add_argument(
    "-i",
    "--input-path",
    dest="input_path",
    help="File path of the csv file to be referenced for extraction",
)
parser.add_argument(
    "-o",
    "--output-path",
    dest="output_path",
    help="Output directory for extracted files",
)
parser.add_argument(
    "-c",
    "--column-name",
    dest="column_name",
    help="Name of column to be extracted",
)
parser.add_argument(
    "-v",
    "--value",
    dest="value",
    help="Value to be extracted",
)
parser.add_argument(
    "--image-path-column-name",
    dest="image_path_column_name",
    default="image_path",
    help="Name of column containing image path",
)


def check_file_existence(file_path: str) -> None:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file '{file_path}' does not exist.")


def check_directory_existence(directory_path: str) -> None:
    if not (os.path.exists(directory_path) and os.path.isdir(directory_path)):
        raise NotADirectoryError(f"The directory '{directory_path}' does not exist.")


def valid_column_name(df: pd.DataFrame, column_name: str) -> None:
    if column_name not in df.columns:
        raise ValueError("Column name does not exist in csv file")
    return


def extract_data(
    df: pd.DataFrame, column_name: str, value: str
) -> typing.Union[None, pd.DataFrame]:
    extract_data = df[df[column_name] == value]
    non_extract_data = df[df[column_name] != value]
    return extract_data, non_extract_data


def output_data(
    output_dir_path: str, df: pd.DataFrame, image_path_column_name: str
) -> None:
    image_path_list: np.ndarray = df[image_path_column_name].to_numpy()

    for image_path in image_path_list:
        image_file_name = os.path.basename(image_path)
        output_file_path = os.path.join(output_dir_path, image_file_name)

        shutil.copy2(image_path, output_file_path)


def main() -> None:

    args = parser.parse_args()
    source_file_path: str = args.input_path
    output_file_path: str = args.output_path
    column_name: str = args.column_name
    value: typing.Union[int, str] = (
        int(args.value) if args.value.isdigit() else args.value
    )
    image_path_column_name: str = args.image_path_column_name

    target_data_path = os.path.join(output_file_path, "target")
    non_target_data_path = os.path.join(output_file_path, "non_target")

    try:
        check_file_existence(source_file_path)
        check_directory_existence(output_file_path)
    except (FileNotFoundError, NotADirectoryError) as e:
        print(e)
        return

    os.mkdir(target_data_path)
    os.mkdir(non_target_data_path)

    try:
        df = pd.read_csv(source_file_path)
        valid_column_name(df, column_name)
    except Exception as e:
        print(e)
        return

    extract_df, non_extract_df = extract_data(df, column_name, value)

    if extract_df.empty:
        print(
            f"There is no data in file {source_file_path} with {column_name} = {value}"
        )
        return

    output_data(
        os.path.join(output_file_path, target_data_path),
        extract_df,
        image_path_column_name,
    )
    output_data(
        os.path.join(output_file_path, non_target_data_path),
        non_extract_df,
        image_path_column_name,
    )


if __name__ == "__main__":
    main()
