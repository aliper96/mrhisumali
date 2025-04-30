#!/usr/bin/env python3
import os
import csv
import json
import argparse
from pathlib import Path
from collections import defaultdict


def main():
    parser = argparse.ArgumentParser(description='Actualizar CSV y JSON basado en archivos tfrecord disponibles')
    parser.add_argument('--csv', default=r"C:\Users\aliha\Desktop\ABYYS\MR.HiSum\dataset\metadata.csv", help='Ruta al archivo CSV')
    parser.add_argument('--json', default=r"C:\Users\aliha\Desktop\ABYYS\MR.HiSum\dataset\mr_hisum_split.json", help='Ruta al archivo JSON con las keys')
    parser.add_argument('--dir', default=r"H:\abyss\youtube8m_data", help='Ruta al directorio con archivos tfrecord')
    parser.add_argument('--output-csv', default="metadata.csv", help='Ruta para guardar el CSV filtrado')
    parser.add_argument('--output-json', default="mr_hisum_split.json", help='Ruta para guardar el JSON actualizado')
    args = parser.parse_args()

    # Obtener lista de archivos tfrecord disponibles en el directorio
    available_files = set()
    for file in os.listdir(args.dir):
        if file.endswith('.tfrecord'):
            # Eliminar extensión .tfrecord
            base_name = os.path.splitext(file)[0]
            available_files.add(base_name)

    print(f"Se encontraron {len(available_files)} archivos tfrecord en el directorio")

    # Cargar datos JSON
    with open(args.json, 'r') as f:
        json_data = json.load(f)

    # Extraer keys del JSON
    train_keys = json_data.get('train_keys', [])
    val_keys = json_data.get('val_keys', [])
    test_keys = json_data.get('test_keys', [])

    print(f"JSON contiene: {len(train_keys)} train_keys, {len(val_keys)} val_keys, {len(test_keys)} test_keys")

    # Leer el CSV
    with open(args.csv, 'r', newline='', encoding='utf-8') as f:
        reader = csv.reader(f)
        header = next(reader)  # Guardar el encabezado
        rows = list(reader)

    # Encontrar el índice de las columnas que necesitamos
    try:
        video_id_index = header.index('video_id')
        yt8m_file_index = header.index('yt8m_file')
    except ValueError as e:
        print(f"Error: El archivo CSV no tiene las columnas requeridas: {e}")
        return

    # Filtrar filas que tienen un archivo tfrecord correspondiente
    filtered_rows = [row for row in rows if row[yt8m_file_index] in available_files]

    print(f"CSV original tenía {len(rows)} registros")
    print(f"Después de filtrar, quedan {len(filtered_rows)} registros")

    # Crear conjuntos de video_ids que permanecen después del filtrado
    remaining_video_ids = set(row[video_id_index] for row in filtered_rows)

    # Actualizar las listas de keys en el JSON
    new_train_keys = [key for key in train_keys if key in remaining_video_ids]
    new_val_keys = [key for key in val_keys if key in remaining_video_ids]
    new_test_keys = [key for key in test_keys if key in remaining_video_ids]

    print(
        f"JSON actualizado tendrá: {len(new_train_keys)} train_keys, {len(new_val_keys)} val_keys, {len(new_test_keys)} test_keys")

    # Crear el nuevo objeto JSON
    new_json_data = {
        'train_keys': new_train_keys,
        'val_keys': new_val_keys,
        'test_keys': new_test_keys
    }

    # Guardar el CSV filtrado
    with open(args.output_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(filtered_rows)

    # Guardar el JSON actualizado
    with open(args.output_json, 'w', encoding='utf-8') as f:
        json.dump(new_json_data, f, indent=2)

    print(f"CSV filtrado guardado en {args.output_csv}")
    print(f"JSON actualizado guardado en {args.output_json}")


if __name__ == "__main__":
    main()